from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from types import SimpleNamespace
from typing import AsyncGenerator, Protocol

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from vocode_contact_center.langchain_support import (
    build_chain,
    build_prompt_inputs_from_messages,
    extract_text_from_langchain_message,
)
from vocode_contact_center.orchestration.models import (
    ConversationOrchestrator,
    ConversationTurnResult,
)
from vocode_contact_center.settings import ContactCenterSettings
from vocode_contact_center.voicebot_graph.adapters.base import (
    AuthenticationAdapter,
    GenesysAdapter,
    SmsSender,
)
from vocode_contact_center.voicebot_graph.intents import classify_global_navigation
from vocode_contact_center.voicebot_graph.state import VoicebotGraphState, clone_state as clone_graph_state
from vocode_contact_center.voicebot_graph.service import (
    VoicebotGraphService,
    VoicebotTurnResult,
)

HYBRID_ROUTER_SYSTEM_PROMPT = """
You route a **banking-only** phone contact center. The freeform assistant is a
**banking** voice agent: it must not act as a general expert on unrelated topics
(self-storage, retail, restaurants, travel, medical or legal advice, etc.).

Choose exactly one action:
- `answer_directly`: use when **mode is generic** and the caller should hear a
  conversational reply, including off-topic **banking-only** steering. When **mode
  is graph**, avoid this action unless the structured-flow context explicitly
  says the step is finished and a generic reply is appropriate.
- `enter_graph_flow`: use when the caller wants a structured workflow: information
  lookup, registration, login/account support, announcements, or feedback/contact
  handling—all **banking** contact center flows.
- `continue_graph`: use when **mode is graph** and the caller is **plausibly**
  continuing the active workflow: answering a requested field (including noisy,
  partial, or split phrases), picking a menu option, confirming a step, or
  repeating or clarifying within the same step. **Default to this** when a field
  is being collected or a structured menu is shown, even if the transcript is
  imperfect.
- `escape_to_generic`: use when **mode is graph** and the caller clearly wants to
  cancel, start over, or leave the workflow without answering the current step
  (e.g. never mind, main menu, stop). When you use this action, provide a brief
  plain-speech `response_text`.

Rules:
- Keep `response_text` empty unless the action is `answer_directly` only when you
  want to bias the downstream banking response, or `escape_to_generic`.
- Prefer `enter_graph_flow` for new structured banking operations when mode is generic.
- Prefer `continue_graph` whenever the structured-flow context shows an active
  collection step or menu, unless the caller is clearly exiting via cancel/main menu.
- Prefer `answer_directly` for off-topic or out-of-domain requests **when mode is
  generic** so the caller gets an explicit **banking-only** correction.
- Follow the **Structured flow context** block when it conflicts with a loose
  reading of the latest utterance.
- Keep any provided `response_text` short, natural, and suitable for live voice.
""".strip()

STICKY_PENDING_AUTH_FIELDS: frozenset[str] = frozenset({"full_name", "phone_number"})
STICKY_ACTIVE_MENUS: frozenset[str] = frozenset(
    {
        "root_intent",
        "interaction_entry",
        "info_selection",
        "change_information",
        "announcements_continue",
        "announcements_terminal",
        "feedback_question",
        "feedback_terminal",
        "registration_terminal",
        "login_terminal",
        "fail_terminal",
    }
)


def _graph_snapshot_is_sticky(snap: VoicebotGraphState | dict[str, object] | None) -> bool:
    if not snap:
        return False
    if snap.get("pending_auth_field") in STICKY_PENDING_AUTH_FIELDS:
        return True
    if snap.get("active_menu") in STICKY_ACTIVE_MENUS:
        return True
    return False


def _structured_flow_context_from_graph(snap: VoicebotGraphState | dict[str, object] | None) -> str:
    if not snap:
        return "No in-memory graph session (caller is not in a committed structured workflow)."
    lines = [
        f"Graph active_menu={snap.get('active_menu')!r}",
        f"Graph pending_auth_field={snap.get('pending_auth_field')!r}",
        f"Graph interaction_context={snap.get('interaction_context')!r}",
        f"Graph current_path={snap.get('current_path')!r}",
    ]
    pending = snap.get("pending_auth_field")
    if pending:
        lines.append(
            "Structured step is active: prefer continue_graph so the graph can interpret the "
            f"utterance against the pending field {pending!r} (including STT noise or partial answers), "
            "unless the caller is clearly asking to cancel or return to the main menu."
        )
    elif snap.get("active_menu") in STICKY_ACTIVE_MENUS:
        lines.append(
            "Structured menu or terminal is active: prefer continue_graph so the graph can "
            "handle the utterance as a menu choice or follow-up unless the caller clearly cancels."
        )
    return "\n".join(lines)


def _apply_sticky_route_override(
    decision: HybridRouteDecision,
    snap: VoicebotGraphState | dict[str, object] | None,
    user_text: str,
) -> HybridRouteDecision:
    if not _graph_snapshot_is_sticky(snap):
        return decision
    navigation = classify_global_navigation(user_text)
    if navigation == "main_menu":
        # Let escape_to_generic use the router's brief wording; only correct answer_directly.
        if decision.action == HybridRouteAction.ANSWER_DIRECTLY:
            return HybridRouteDecision(action=HybridRouteAction.CONTINUE_GRAPH)
        return decision
    if decision.action == HybridRouteAction.ANSWER_DIRECTLY:
        return HybridRouteDecision(action=HybridRouteAction.CONTINUE_GRAPH)
    return decision


class HybridRouteAction(str, Enum):
    ANSWER_DIRECTLY = "answer_directly"
    ENTER_GRAPH_FLOW = "enter_graph_flow"
    CONTINUE_GRAPH = "continue_graph"
    ESCAPE_TO_GENERIC = "escape_to_generic"


class HybridRouteDecision(BaseModel):
    action: HybridRouteAction
    response_text: str = Field(
        default="",
        description="Optional plain-speech response text for escape transitions or direct answers.",
    )


@dataclass
class HybridSessionState:
    session_id: str
    call_context: str
    metadata: dict[str, str] = field(default_factory=dict)
    history: list[tuple[str, str]] = field(default_factory=list)
    mode: str = "generic"
    last_graph_outcome: str | None = None
    last_graph_menu: str | None = None


class HybridRoutePolicy(Protocol):
    async def decide(
        self,
        *,
        state: HybridSessionState,
        latest_user_input: str,
        structured_flow_context: str = "",
    ) -> HybridRouteDecision:
        ...


class GenericConversationResponder(Protocol):
    async def respond(
        self,
        *,
        state: HybridSessionState,
        latest_user_input: str,
        response_prefix: str = "",
    ) -> str:
        ...


class LangChainHybridRoutePolicy:
    def __init__(self, settings: ContactCenterSettings):
        model = init_chat_model(
            model=settings.langchain_model_name,
            model_provider=settings.langchain_provider,
            temperature=settings.langchain_temperature,
            max_tokens=settings.langchain_max_tokens,
        )
        if not hasattr(model, "with_structured_output"):
            raise ValueError(
                "The configured chat model does not support structured outputs for "
                "hybrid orchestration routing."
            )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", HYBRID_ROUTER_SYSTEM_PROMPT),
                (
                    "system",
                    "Current mode: {mode}\n"
                    "Active graph menu: {active_graph_menu}\n"
                    "Last graph outcome: {last_graph_outcome}\n"
                    "Allowed actions: {allowed_actions}\n"
                    "Structured flow context:\n{structured_flow_context}",
                ),
                ("system", "{call_context}"),
                ("system", "{conversation_summary}"),
                ("placeholder", "{chat_history}"),
                ("human", "Latest caller message: {latest_user_input}"),
            ]
        )
        self._chain = prompt | model.with_structured_output(HybridRouteDecision)
        self._settings = settings

    async def decide(
        self,
        *,
        state: HybridSessionState,
        latest_user_input: str,
        structured_flow_context: str = "",
    ) -> HybridRouteDecision:
        prompt_inputs = build_prompt_inputs_from_messages(
            state.history,
            call_context=state.call_context,
            recent_message_limit=self._settings.langchain_recent_message_limit,
            summary_max_messages=self._settings.langchain_summary_max_messages,
            summary_max_chars=self._settings.langchain_summary_max_chars,
        )
        allowed_actions = (
            ", ".join(
                [
                    HybridRouteAction.ANSWER_DIRECTLY.value,
                    HybridRouteAction.CONTINUE_GRAPH.value,
                    HybridRouteAction.ESCAPE_TO_GENERIC.value,
                ]
            )
            if state.mode == "graph"
            else ", ".join(
                [
                    HybridRouteAction.ANSWER_DIRECTLY.value,
                    HybridRouteAction.ENTER_GRAPH_FLOW.value,
                ]
            )
        )
        result = await self._chain.ainvoke(
            {
                **prompt_inputs,
                "mode": state.mode,
                "active_graph_menu": state.last_graph_menu or "none",
                "last_graph_outcome": state.last_graph_outcome or "none",
                "allowed_actions": allowed_actions,
                "structured_flow_context": structured_flow_context or "(not provided)",
                "latest_user_input": latest_user_input,
            }
        )
        if isinstance(result, HybridRouteDecision):
            return result
        return HybridRouteDecision.model_validate(result)


class LangChainGenericConversationResponder:
    def __init__(self, settings: ContactCenterSettings):
        self._settings = settings
        self._chain = build_chain(
            SimpleNamespace(
                prompt_preamble=settings.agent_prompt_preamble,
                model_name=settings.langchain_model_name,
                provider=settings.langchain_provider,
                temperature=settings.langchain_temperature,
                max_tokens=settings.langchain_max_tokens,
            )
        )

    async def respond(
        self,
        *,
        state: HybridSessionState,
        latest_user_input: str,
        response_prefix: str = "",
    ) -> str:
        messages = list(state.history)
        combined_input = latest_user_input.strip()
        if response_prefix:
            combined_input = f"{response_prefix.strip()} Caller request: {combined_input}".strip()
        messages.append(("human", combined_input))
        prompt_inputs = build_prompt_inputs_from_messages(
            messages,
            call_context=state.call_context,
            recent_message_limit=self._settings.langchain_recent_message_limit,
            summary_max_messages=self._settings.langchain_summary_max_messages,
            summary_max_chars=self._settings.langchain_summary_max_chars,
        )
        result = await self._chain.ainvoke(prompt_inputs)
        return extract_text_from_langchain_message(result)


class HybridConversationOrchestratorService(ConversationOrchestrator):
    def __init__(
        self,
        settings: ContactCenterSettings,
        *,
        route_policy: HybridRoutePolicy | None = None,
        generic_responder: GenericConversationResponder | None = None,
        graph_service: VoicebotGraphService | None = None,
        auth_adapter: AuthenticationAdapter | None = None,
        genesys_adapter: GenesysAdapter | None = None,
        sms_sender: SmsSender | None = None,
    ) -> None:
        self.settings = settings
        self.route_policy = route_policy or LangChainHybridRoutePolicy(settings)
        self.generic_responder = generic_responder or LangChainGenericConversationResponder(
            settings
        )
        self.graph_service = graph_service or VoicebotGraphService(
            settings,
            auth_adapter=auth_adapter,
            genesys_adapter=genesys_adapter,
            sms_sender=sms_sender,
        )
        self._sessions: dict[str, HybridSessionState] = {}

    async def _route_decision(
        self,
        *,
        state: HybridSessionState,
        user_text: str,
    ) -> HybridRouteDecision:
        graph_snap = self.graph_service.export_session(state.session_id)
        ctx = _structured_flow_context_from_graph(graph_snap)
        decision = await self.route_policy.decide(
            state=state,
            latest_user_input=user_text,
            structured_flow_context=ctx,
        )
        return _apply_sticky_route_override(decision, graph_snap, user_text)

    async def run_turn(
        self,
        session_id: str,
        user_text: str,
        *,
        call_context: str,
        metadata: dict[str, str] | None = None,
        commit: bool = True,
    ) -> ConversationTurnResult:
        state = self._prepare_state(
            session_id=session_id,
            call_context=call_context,
            metadata=metadata,
        )
        decision = await self._route_decision(state=state, user_text=user_text)

        if state.mode == "graph":
            if decision.action == HybridRouteAction.ESCAPE_TO_GENERIC:
                return self._build_escape_result(
                    state=state,
                    user_text=user_text,
                    response_text=decision.response_text,
                    commit=commit,
                )
            if decision.action == HybridRouteAction.ANSWER_DIRECTLY:
                return await self._build_generic_answer_result(
                    state=state,
                    user_text=user_text,
                    response_prefix=decision.response_text,
                    commit=commit,
                )
            return await self._delegate_to_graph(
                state=state,
                user_text=user_text,
                call_context=call_context,
                metadata=metadata,
                commit=commit,
            )

        if decision.action == HybridRouteAction.ENTER_GRAPH_FLOW:
            return await self._delegate_to_graph(
                state=state,
                user_text=user_text,
                call_context=call_context,
                metadata=metadata,
                commit=commit,
            )

        return await self._build_generic_answer_result(
            state=state,
            user_text=user_text,
            response_prefix=decision.response_text,
            commit=commit,
        )

    async def preview_turn(
        self,
        session_id: str,
        user_text: str,
        *,
        call_context: str,
        metadata: dict[str, str] | None = None,
    ) -> ConversationTurnResult:
        return await self.run_turn(
            session_id,
            user_text,
            call_context=call_context,
            metadata=metadata,
            commit=False,
        )

    async def stream_text_response(self, text: str) -> AsyncGenerator[str, None]:
        for token in _text_to_tokens(text):
            yield token

    async def stream_generate_response(
        self,
        session_id: str,
        user_text: str,
        *,
        call_context: str,
        metadata: dict[str, str] | None = None,
        commit: bool = True,
    ) -> AsyncGenerator[str, None]:
        state = self._prepare_state(
            session_id=session_id,
            call_context=call_context,
            metadata=metadata,
        )
        decision = await self._route_decision(state=state, user_text=user_text)

        if state.mode == "graph":
            if decision.action == HybridRouteAction.ESCAPE_TO_GENERIC:
                text = decision.response_text.strip() or (
                    "Sure, we can step out of that. How else can I help?"
                )
                async for tok in self.stream_text_response(text):
                    yield tok
                if commit:
                    next_state = deepcopy(state)
                    next_state.mode = "generic"
                    next_state.last_graph_menu = None
                    self.graph_service.clear_session(state.session_id)
                    self._commit_state(next_state, user_text=user_text, response_text=text)
                return

            if decision.action == HybridRouteAction.ANSWER_DIRECTLY:
                response_text = await self.generic_responder.respond(
                    state=state,
                    latest_user_input=user_text,
                    response_prefix=decision.response_text,
                )
                async for tok in self.stream_text_response(response_text):
                    yield tok
                if commit:
                    next_state = deepcopy(state)
                    next_state.mode = "generic"
                    next_state.last_graph_menu = None
                    self.graph_service.clear_session(state.session_id)
                    self._commit_state(
                        next_state,
                        user_text=user_text,
                        response_text=response_text,
                    )
                return

            async for tok in self.graph_service.stream_generate_response(
                state.session_id,
                user_text,
                call_context=call_context,
                metadata=metadata,
                commit=commit,
            ):
                yield tok
            if commit:
                self._commit_hybrid_after_graph_stream(
                    state=state,
                    user_text=user_text,
                    session_id=session_id,
                )
            return

        if decision.action == HybridRouteAction.ENTER_GRAPH_FLOW:
            async for tok in self.graph_service.stream_generate_response(
                session_id,
                user_text,
                call_context=call_context,
                metadata=metadata,
                commit=commit,
            ):
                yield tok
            if commit:
                self._commit_hybrid_after_graph_stream(
                    state=state,
                    user_text=user_text,
                    session_id=session_id,
                )
            return

        response_text = await self.generic_responder.respond(
            state=state,
            latest_user_input=user_text,
            response_prefix=decision.response_text,
        )
        async for tok in self.stream_text_response(response_text):
            yield tok
        if commit:
            next_state = deepcopy(state)
            next_state.mode = "generic"
            self._commit_state(next_state, user_text=user_text, response_text=response_text)

    def _commit_hybrid_after_graph_stream(
        self,
        *,
        state: HybridSessionState,
        user_text: str,
        session_id: str,
    ) -> None:
        snap = self.graph_service.export_session(session_id)
        if not snap:
            graph_result = VoicebotTurnResult(
                text="",
                final_outcome=None,
                active_menu=None,
                menu_options=[],
                artifacts={},
                adapter_results={},
                state_snapshot={},
            )
        else:
            graph_result = VoicebotTurnResult(
                text=snap.get("response_text") or "",
                final_outcome=snap.get("final_outcome"),
                active_menu=snap.get("active_menu"),
                menu_options=list(snap.get("menu_options", [])),
                artifacts=dict(snap.get("artifacts", {})),
                adapter_results=dict(snap.get("adapter_results", {})),
                state_snapshot=clone_graph_state(snap),
            )
        next_state = deepcopy(state)
        next_state.last_graph_outcome = graph_result.final_outcome
        next_state.last_graph_menu = _active_graph_menu(graph_result)
        next_state.mode = "graph" if _graph_result_is_active(graph_result) else "generic"
        if next_state.mode == "generic":
            self.graph_service.clear_session(state.session_id)
        self._commit_state(
            next_state,
            user_text=user_text,
            response_text=graph_result.text,
        )

    def _prepare_state(
        self,
        *,
        session_id: str,
        call_context: str,
        metadata: dict[str, str] | None,
    ) -> HybridSessionState:
        existing = self._sessions.get(session_id)
        if existing is None:
            return HybridSessionState(
                session_id=session_id,
                call_context=call_context,
                metadata=dict(metadata or {}),
            )

        state = deepcopy(existing)
        state.call_context = call_context
        if metadata:
            merged = dict(state.metadata)
            merged.update(metadata)
            state.metadata = merged
        return state

    async def _build_generic_answer_result(
        self,
        *,
        state: HybridSessionState,
        user_text: str,
        response_prefix: str = "",
        commit: bool,
    ) -> ConversationTurnResult:
        response_text = await self.generic_responder.respond(
            state=state,
            latest_user_input=user_text,
            response_prefix=response_prefix,
        )
        next_state = deepcopy(state)
        next_state.mode = "generic"
        next_state.last_graph_menu = None
        if commit:
            self.graph_service.clear_session(state.session_id)
            self._commit_state(next_state, user_text=user_text, response_text=response_text)
        return self._build_generic_result(next_state, response_text)

    def _build_escape_result(
        self,
        *,
        state: HybridSessionState,
        user_text: str,
        response_text: str,
        commit: bool,
    ) -> ConversationTurnResult:
        next_state = deepcopy(state)
        next_state.mode = "generic"
        next_state.last_graph_menu = None
        text = response_text.strip() or "Sure, we can step out of that. How else can I help?"
        if commit:
            self.graph_service.clear_session(state.session_id)
            self._commit_state(next_state, user_text=user_text, response_text=text)
        return self._build_generic_result(next_state, text)

    async def _delegate_to_graph(
        self,
        *,
        state: HybridSessionState,
        user_text: str,
        call_context: str,
        metadata: dict[str, str] | None,
        commit: bool,
    ) -> ConversationTurnResult:
        graph_result = (
            await self.graph_service.run_turn(
                state.session_id,
                user_text,
                call_context=call_context,
                metadata=metadata,
                commit=True,
            )
            if commit
            else await self.graph_service.preview_turn(
                state.session_id,
                user_text,
                call_context=call_context,
                metadata=metadata,
            )
        )
        next_state = deepcopy(state)
        next_state.last_graph_outcome = graph_result.final_outcome
        next_state.last_graph_menu = _active_graph_menu(graph_result)
        next_state.mode = "graph" if _graph_result_is_active(graph_result) else "generic"
        if commit:
            if next_state.mode == "generic":
                self.graph_service.clear_session(state.session_id)
            self._commit_state(next_state, user_text=user_text, response_text=graph_result.text)
        return self._build_graph_result(next_state, graph_result)

    def _commit_state(
        self,
        state: HybridSessionState,
        *,
        user_text: str,
        response_text: str,
    ) -> None:
        if user_text.strip():
            state.history.append(("human", user_text.strip()))
        if response_text.strip():
            state.history.append(("ai", response_text.strip()))
        self._sessions[state.session_id] = deepcopy(state)

    def _build_generic_result(
        self,
        state: HybridSessionState,
        text: str,
    ) -> ConversationTurnResult:
        return ConversationTurnResult(
            text=text,
            final_outcome=None,
            active_menu=None,
            menu_options=[],
            artifacts={},
            adapter_results={},
            state_snapshot=self._hybrid_state_snapshot(state),
        )

    def _build_graph_result(
        self,
        state: HybridSessionState,
        graph_result: VoicebotTurnResult,
    ) -> ConversationTurnResult:
        snapshot = dict(graph_result.state_snapshot)
        snapshot.update(self._hybrid_state_snapshot(state))
        return ConversationTurnResult(
            text=graph_result.text,
            final_outcome=graph_result.final_outcome,
            active_menu=_active_graph_menu(graph_result),
            menu_options=list(graph_result.menu_options),
            artifacts=dict(graph_result.artifacts),
            adapter_results=dict(graph_result.adapter_results),
            state_snapshot=snapshot,
        )

    def _hybrid_state_snapshot(self, state: HybridSessionState) -> dict[str, object]:
        return {
            "hybrid_mode": state.mode,
            "hybrid_history": list(state.history),
            "hybrid_last_graph_outcome": state.last_graph_outcome,
            "hybrid_last_graph_menu": state.last_graph_menu,
            "metadata": dict(state.metadata),
            "call_context": state.call_context,
        }


def _active_graph_menu(result: VoicebotTurnResult) -> str | None:
    if result.active_menu:
        return result.active_menu
    pending_auth_field = result.state_snapshot.get("pending_auth_field")
    if pending_auth_field:
        return "authentication"
    return None


def _graph_result_is_active(result: VoicebotTurnResult) -> bool:
    return _active_graph_menu(result) is not None


def _text_to_tokens(text: str) -> list[str]:
    if not text:
        return []
    return re.findall(r"\S+\s*", text)
