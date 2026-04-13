from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any, AsyncGenerator, Protocol

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

from vocode_contact_center.langchain_support import build_prompt_inputs_from_messages
from vocode_contact_center.orchestration.models import (
    ConversationIntent,
    ConversationPolicyDecision,
    ConversationSessionState,
    ConversationStage,
    ConversationTurnResult,
    InteractionContext,
    PolicyAction,
)
from vocode_contact_center.orchestration.prompts import (
    DEFAULT_ORCHESTRATOR_SYSTEM_PROMPT,
    STAGE_GUIDANCE,
)
from vocode_contact_center.sms import build_registration_confirmation_message
from vocode_contact_center.settings import ContactCenterSettings
from vocode_contact_center.voicebot_graph.adapters.base import (
    AuthenticationAdapter,
    AuthenticationRequest,
    GenesysAdapter,
    GenesysRequest,
    SmsRequest,
    SmsResult,
    SmsSender,
)
from vocode_contact_center.voicebot_graph.adapters.stub import (
    StubAuthenticationAdapter,
    StubGenesysAdapter,
    StubSmsSender,
)
from vocode_contact_center.voicebot_graph.nodes.terminals import (
    information_products_response,
    information_store_response,
    terminal_response_text,
)

ROOT_OPTIONS = [
    "information",
    "store_information",
    "product_information",
    "registration",
    "login",
    "announcements",
    "feedback",
    "general_support",
]


class ConversationPolicy(Protocol):
    async def decide(
        self,
        *,
        state: ConversationSessionState,
        available_actions: list[str],
        available_options: list[str],
    ) -> ConversationPolicyDecision:
        ...


class LangChainConversationPolicy:
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
                "LLM-led orchestration."
            )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", DEFAULT_ORCHESTRATOR_SYSTEM_PROMPT),
                (
                    "system",
                    "Current stage: {stage}\n"
                    "Stage guidance: {stage_guidance}\n"
                    "Allowed actions: {available_actions}\n"
                    "Allowed options: {available_options}\n"
                    "Current session state JSON: {session_state_json}",
                ),
                ("system", "{call_context}"),
                ("system", "{conversation_summary}"),
                ("placeholder", "{chat_history}"),
                ("human", "Latest caller message: {latest_user_input}"),
            ]
        )
        self._chain = prompt | model.with_structured_output(ConversationPolicyDecision)
        self._settings = settings

    async def decide(
        self,
        *,
        state: ConversationSessionState,
        available_actions: list[str],
        available_options: list[str],
    ) -> ConversationPolicyDecision:
        prompt_inputs = build_prompt_inputs_from_messages(
            state.history,
            call_context=state.call_context,
            recent_message_limit=self._settings.langchain_recent_message_limit,
            summary_max_messages=self._settings.langchain_summary_max_messages,
            summary_max_chars=self._settings.langchain_summary_max_chars,
        )
        result = await self._chain.ainvoke(
            {
                **prompt_inputs,
                "stage": state.stage.value,
                "stage_guidance": STAGE_GUIDANCE.get(state.stage.value, ""),
                "available_actions": ", ".join(available_actions),
                "available_options": ", ".join(available_options) if available_options else "none",
                "session_state_json": json.dumps(
                    state.model_dump(mode="json", exclude={"history"}),
                    ensure_ascii=True,
                    sort_keys=True,
                ),
                "latest_user_input": state.latest_user_input,
            }
        )
        if isinstance(result, ConversationPolicyDecision):
            return result
        return ConversationPolicyDecision.model_validate(result)


class LLMConversationOrchestratorService:
    def __init__(
        self,
        settings: ContactCenterSettings,
        *,
        policy: ConversationPolicy | None = None,
        auth_adapter: AuthenticationAdapter | None = None,
        genesys_adapter: GenesysAdapter | None = None,
        sms_sender: SmsSender | None = None,
    ) -> None:
        self.settings = settings
        self.policy = policy or LangChainConversationPolicy(settings)
        self.auth_adapter = auth_adapter or StubAuthenticationAdapter(settings)
        self.genesys_adapter = genesys_adapter or StubGenesysAdapter()
        self.sms_sender = sms_sender or StubSmsSender()
        self._sessions: dict[str, ConversationSessionState] = {}

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
            session_id,
            user_text,
            call_context=call_context,
            metadata=metadata,
        )
        next_state = await self._handle_turn(state)
        if commit:
            committed_state = deepcopy(next_state)
            self._append_history(committed_state)
            self._sessions[session_id] = committed_state
        return self._build_turn_result(next_state)

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
        result = await self.run_turn(
            session_id,
            user_text,
            call_context=call_context,
            metadata=metadata,
            commit=commit,
        )
        async for token in self.stream_text_response(result.text):
            yield token

    def _prepare_state(
        self,
        session_id: str,
        user_text: str,
        *,
        call_context: str,
        metadata: dict[str, str] | None,
    ) -> ConversationSessionState:
        existing = self._sessions.get(session_id)
        if existing is None:
            state = ConversationSessionState(
                session_id=session_id,
                call_context=call_context,
                metadata=dict(metadata or {}),
            )
        else:
            state = deepcopy(existing)
            if state.final_outcome:
                state.final_outcome = None
                state.artifacts = {}
                state.adapter_results = {}
                state.genesys_requested = False
        state.call_context = call_context
        if metadata:
            merged = dict(state.metadata)
            merged.update(metadata)
            state.metadata = merged
        state.latest_user_input = user_text.strip()
        state.response_text = ""
        state.adapter_results = {}
        state.artifacts = {}
        return state

    async def _handle_turn(self, state: ConversationSessionState) -> ConversationSessionState:
        if state.stage == ConversationStage.AUTHENTICATION:
            return await self._handle_authentication_turn(state)

        available_actions, available_options = self._policy_surface(state)
        decision = await self.policy.decide(
            state=state,
            available_actions=available_actions,
            available_options=available_options,
        )
        return await self._apply_policy_decision(state, decision, available_options)

    def _policy_surface(self, state: ConversationSessionState) -> tuple[list[str], list[str]]:
        default_actions = [
            PolicyAction.SELECT_OPTION.value,
            PolicyAction.ASK_CLARIFYING_QUESTION.value,
            PolicyAction.FALLBACK.value,
        ]
        stage_options = {
            ConversationStage.ROOT: ROOT_OPTIONS,
            ConversationStage.INFORMATION: ["store", "products", "other"],
            ConversationStage.REGISTRATION_TERMINAL: [
                "perform_registration",
                "registration_sms_confirmation",
                "generic_sms",
            ],
            ConversationStage.LOGIN_TERMINAL: [
                "perform_login",
                "update_balance",
                "details",
            ],
            ConversationStage.FAIL_TERMINAL: [
                "communication",
                "generic_sms",
                "details",
            ],
            ConversationStage.ANNOUNCEMENTS_CONTINUE: ["continue", "stop"],
            ConversationStage.ANNOUNCEMENTS_TERMINAL: ["human_agent", "call_back"],
            ConversationStage.FEEDBACK_QUESTION: ["back_to_chat", "genesys"],
            ConversationStage.FEEDBACK_TERMINAL: ["human_agent", "contact"],
        }
        actions = list(default_actions)
        if state.stage == ConversationStage.ROOT:
            actions.insert(1, PolicyAction.ANSWER_DIRECTLY.value)
        return actions, stage_options.get(state.stage, [])

    async def _apply_policy_decision(
        self,
        state: ConversationSessionState,
        decision: ConversationPolicyDecision,
        available_options: list[str],
    ) -> ConversationSessionState:
        if decision.action == PolicyAction.ANSWER_DIRECTLY:
            return self._complete(
                state,
                response_text=decision.response_text
                or "Of course. Let me help with that.",
                final_outcome="general_answer",
            )

        if decision.action in {PolicyAction.ASK_CLARIFYING_QUESTION, PolicyAction.FALLBACK}:
            return self._stay_in_stage_with_prompt(
                state,
                response_text=decision.response_text or self._default_prompt_for_stage(state),
                available_options=available_options,
            )

        if decision.action != PolicyAction.SELECT_OPTION:
            return self._stay_in_stage_with_prompt(
                state,
                response_text=self._default_prompt_for_stage(state),
                available_options=available_options,
            )

        selected_option = (decision.selected_option or "").strip().lower()
        if selected_option not in available_options:
            return self._stay_in_stage_with_prompt(
                state,
                response_text=decision.response_text or self._default_prompt_for_stage(state),
                available_options=available_options,
            )

        if state.stage == ConversationStage.ROOT:
            return await self._apply_root_selection(state, decision, selected_option)
        if state.stage == ConversationStage.INFORMATION:
            return self._apply_information_selection(state, decision, selected_option)
        if state.stage in {
            ConversationStage.REGISTRATION_TERMINAL,
            ConversationStage.LOGIN_TERMINAL,
            ConversationStage.FAIL_TERMINAL,
            ConversationStage.ANNOUNCEMENTS_TERMINAL,
            ConversationStage.FEEDBACK_TERMINAL,
        }:
            return self._complete(
                state,
                response_text=terminal_response_text(state.stage.value, selected_option),
                final_outcome=selected_option,
            )
        if state.stage == ConversationStage.ANNOUNCEMENTS_CONTINUE:
            return await self._apply_announcements_continue_selection(
                state,
                decision,
                selected_option,
            )
        if state.stage == ConversationStage.FEEDBACK_QUESTION:
            return await self._apply_feedback_selection(state, decision, selected_option)
        return self._stay_in_stage_with_prompt(
            state,
            response_text=self._default_prompt_for_stage(state),
            available_options=available_options,
        )

    async def _apply_root_selection(
        self,
        state: ConversationSessionState,
        decision: ConversationPolicyDecision,
        selected_option: str,
    ) -> ConversationSessionState:
        if selected_option == "information":
            state.stage = ConversationStage.INFORMATION
            state.current_intent = ConversationIntent.INFORMATION
            state.available_options = ["store", "products", "other"]
            state.response_text = self._safe_response_text(
                decision.response_text,
                "I can help with store information, product information, or something else. "
                "Which would you like?",
            )
            return state
        if selected_option == "store_information":
            response_text, artifacts = information_store_response(self.settings)
            state.current_intent = ConversationIntent.INFORMATION
            return self._complete(
                state,
                response_text=self._safe_response_text(
                    decision.response_text,
                    response_text,
                    required_fragments=(self.settings.information_store_website_url,),
                ),
                final_outcome="website",
                artifacts=artifacts,
            )
        if selected_option == "product_information":
            response_text, artifacts = information_products_response(self.settings)
            state.current_intent = ConversationIntent.INFORMATION
            return self._complete(
                state,
                response_text=self._safe_response_text(
                    decision.response_text,
                    response_text,
                    required_fragments=(self.settings.information_products_pdf_url,),
                ),
                final_outcome="pdf",
                artifacts=artifacts,
            )
        if selected_option == "registration":
            state.current_intent = ConversationIntent.INTERACTION
            return await self._begin_authentication_flow(state, InteractionContext.REGISTRATION)
        if selected_option == "login":
            state.current_intent = ConversationIntent.INTERACTION
            return await self._begin_authentication_flow(state, InteractionContext.LOGIN)
        if selected_option == "announcements":
            state.stage = ConversationStage.ANNOUNCEMENTS_CONTINUE
            state.current_intent = ConversationIntent.ANNOUNCEMENTS
            state.available_options = ["continue", "stop"]
            state.response_text = self._safe_response_text(
                decision.response_text,
                (
                    f"{self.settings.announcements_message} If you'd like, I can also connect "
                    "you to contact center support after that. Would you like me to continue?"
                ),
            )
            return state
        if selected_option == "feedback":
            state.stage = ConversationStage.FEEDBACK_QUESTION
            state.current_intent = ConversationIntent.FEEDBACK
            state.available_options = ["back_to_chat", "genesys"]
            state.response_text = self._safe_response_text(
                decision.response_text,
                self.settings.feedback_question_prompt,
            )
            return state
        if selected_option == "general_support":
            return self._complete(
                state,
                response_text=decision.response_text
                or "Of course. Tell me a little more and I'll do my best to help.",
                final_outcome="general_support",
            )
        return self._stay_in_stage_with_prompt(
            state,
            response_text=self._default_prompt_for_stage(state),
            available_options=ROOT_OPTIONS,
        )

    def _apply_information_selection(
        self,
        state: ConversationSessionState,
        decision: ConversationPolicyDecision,
        selected_option: str,
    ) -> ConversationSessionState:
        if selected_option == "store":
            response_text, artifacts = information_store_response(self.settings)
            return self._complete(
                state,
                response_text=self._safe_response_text(
                    decision.response_text,
                    response_text,
                    required_fragments=(self.settings.information_store_website_url,),
                ),
                final_outcome="website",
                artifacts=artifacts,
            )
        if selected_option == "products":
            response_text, artifacts = information_products_response(self.settings)
            return self._complete(
                state,
                response_text=self._safe_response_text(
                    decision.response_text,
                    response_text,
                    required_fragments=(self.settings.information_products_pdf_url,),
                ),
                final_outcome="pdf",
                artifacts=artifacts,
            )
        return self._stay_in_stage_with_prompt(
            state,
            response_text=self._safe_response_text(
                decision.response_text,
                "No problem. Tell me whether you need store information, product information, "
                "or a different kind of general information.",
            ),
            available_options=["store", "products", "other"],
        )

    async def _apply_announcements_continue_selection(
        self,
        state: ConversationSessionState,
        decision: ConversationPolicyDecision,
        selected_option: str,
    ) -> ConversationSessionState:
        if selected_option == "stop":
            return self._complete(
                state,
                response_text=self._safe_response_text(
                    decision.response_text,
                    "Of course. I'll leave it there after the announcements.",
                ),
                final_outcome="announcements_stopped",
            )
        return await self._connect_genesys(
            state,
            path_name="announcements",
            next_stage=ConversationStage.ANNOUNCEMENTS_TERMINAL,
            next_options=["human_agent", "call_back"],
            fallback_prompt=(
                "I can connect you to a human agent, or I can arrange a call back instead. "
                "Which works better for you?"
            ),
            preferred_text=decision.response_text,
        )

    async def _apply_feedback_selection(
        self,
        state: ConversationSessionState,
        decision: ConversationPolicyDecision,
        selected_option: str,
    ) -> ConversationSessionState:
        if selected_option == "back_to_chat":
            return self._complete(
                state,
                response_text=self._safe_response_text(
                    decision.response_text,
                    "Absolutely. I'll send you back to the chat now.",
                ),
                final_outcome="back_to_chat",
            )
        return await self._connect_genesys(
            state,
            path_name="feedback",
            next_stage=ConversationStage.FEEDBACK_TERMINAL,
            next_options=["human_agent", "contact"],
            fallback_prompt=(
                "I can connect you with a human agent, or I can submit a contact request for "
                "follow-up. Which would you prefer?"
            ),
            preferred_text=decision.response_text,
        )

    async def _begin_authentication_flow(
        self,
        state: ConversationSessionState,
        context: InteractionContext,
    ) -> ConversationSessionState:
        state.stage = ConversationStage.AUTHENTICATION
        state.interaction_context = context
        state.available_options = []
        return await self._run_authentication(state)

    async def _handle_authentication_turn(
        self,
        state: ConversationSessionState,
    ) -> ConversationSessionState:
        if not state.pending_auth_field:
            return await self._run_authentication(state)
        normalized_value = state.latest_user_input.strip()
        if not normalized_value:
            state.response_text = (
                "I still need that detail before I can continue. Take your time."
            )
            return state
        updated_data = dict(state.collected_data)
        updated_data[state.pending_auth_field] = normalized_value
        state.collected_data = updated_data
        state.pending_auth_field = None
        return await self._run_authentication(state)

    async def _run_authentication(
        self,
        state: ConversationSessionState,
    ) -> ConversationSessionState:
        request = AuthenticationRequest(
            session_id=state.session_id,
            call_context=state.call_context,
            interaction_context=state.interaction_context.value if state.interaction_context else None,
            latest_user_input=state.latest_user_input,
            collected_data=dict(state.collected_data),
            auth_attempts=state.auth_attempts,
        )
        result = await self.auth_adapter.authenticate(request)
        state.auth_attempts += 1
        state.adapter_results = {
            "authentication": {
                "status": result.status,
                "metadata": result.metadata,
            }
        }
        if result.normalized_data:
            updated_data = dict(state.collected_data)
            updated_data.update(result.normalized_data)
            state.collected_data = updated_data

        if result.status == "needs_customer_input":
            state.pending_auth_field = result.requested_field
            state.response_text = result.prompt
            return state

        if result.status == "needs_sms_confirmation":
            sms_result = await self._send_registration_confirmation_sms(state)
            state.adapter_results["sms"] = {
                "status": sms_result.status,
                "metadata": {
                    **sms_result.metadata,
                    **(
                        {"provider_message_id": sms_result.provider_message_id}
                        if sms_result.provider_message_id
                        else {}
                    ),
                    **(
                        {"error_message": sms_result.error_message}
                        if sms_result.error_message
                        else {}
                    ),
                },
            }
            if sms_result.status == "sent":
                updated_data = dict(state.collected_data)
                updated_data["sms_confirmed"] = "true"
                state.collected_data = updated_data
                state.artifacts = {"sms_status": "sent"}
                if sms_result.provider_message_id:
                    state.artifacts["sms_message_id"] = sms_result.provider_message_id
                response_prefix = "I've sent the SMS confirmation step through, so we can keep moving. "
            else:
                state.artifacts = {"sms_status": "failed"}
                response_prefix = (
                    "I couldn't send the SMS confirmation just yet, but I can still help with the next step here. "
                )
            return self._move_to_terminal_stage(
                state,
                response_text=response_prefix
                + self._terminal_prompt_for_context(state.interaction_context),
            )

        if result.status == "success":
            return self._move_to_terminal_stage(
                state,
                response_text=self._terminal_prompt_for_context(state.interaction_context),
            )

        state.stage = ConversationStage.FAIL_TERMINAL
        state.available_options = ["communication", "generic_sms", "details"]
        state.response_text = (
            "It looks like authentication didn't fully complete. I can still help with "
            "general communication options, send an SMS, or share general details. "
            "Which would you like?"
        )
        return state

    def _move_to_terminal_stage(
        self,
        state: ConversationSessionState,
        *,
        response_text: str,
    ) -> ConversationSessionState:
        if state.interaction_context == InteractionContext.REGISTRATION:
            state.stage = ConversationStage.REGISTRATION_TERMINAL
            state.available_options = [
                "perform_registration",
                "registration_sms_confirmation",
                "generic_sms",
            ]
        else:
            state.stage = ConversationStage.LOGIN_TERMINAL
            state.available_options = ["perform_login", "update_balance", "details"]
        state.response_text = response_text
        return state

    async def _send_registration_confirmation_sms(
        self,
        state: ConversationSessionState,
    ) -> SmsResult:
        phone_number = state.collected_data.get("phone_number", "").strip()
        if not phone_number:
            return SmsResult(
                status="failed",
                error_message="No phone number was available for the confirmation SMS.",
                metadata={"provider": "application"},
            )
        return await self.sms_sender.send(
            SmsRequest(
                session_id=state.session_id,
                recipient_phone_number=phone_number,
                message=build_registration_confirmation_message(
                    self.settings,
                    state.collected_data,
                ),
                context="registration_confirmation",
                metadata=dict(state.metadata),
            )
        )

    async def _connect_genesys(
        self,
        state: ConversationSessionState,
        *,
        path_name: str,
        next_stage: ConversationStage,
        next_options: list[str],
        fallback_prompt: str,
        preferred_text: str,
    ) -> ConversationSessionState:
        result = await self.genesys_adapter.connect(
            GenesysRequest(
                session_id=state.session_id,
                path_name=path_name,
                latest_user_input=state.latest_user_input,
                metadata=dict(state.metadata),
            )
        )
        state.genesys_requested = True
        state.stage = next_stage
        state.available_options = next_options
        state.adapter_results = {
            "genesys": {
                "status": result.status,
                "metadata": result.metadata,
            }
        }
        state.response_text = self._safe_response_text(preferred_text, fallback_prompt)
        return state

    def _stay_in_stage_with_prompt(
        self,
        state: ConversationSessionState,
        *,
        response_text: str,
        available_options: list[str],
    ) -> ConversationSessionState:
        state.response_text = response_text
        state.available_options = available_options
        return state

    def _complete(
        self,
        state: ConversationSessionState,
        *,
        response_text: str,
        final_outcome: str,
        artifacts: dict[str, str] | None = None,
    ) -> ConversationSessionState:
        state.response_text = response_text
        state.final_outcome = final_outcome
        state.last_completed_path = state.current_intent.value if state.current_intent else None
        state.stage = ConversationStage.ROOT
        state.available_options = []
        state.pending_auth_field = None
        state.collected_data = {}
        state.interaction_context = None
        state.current_intent = None
        state.artifacts = dict(artifacts or {})
        state.genesys_requested = False
        return state

    def _build_turn_result(self, state: ConversationSessionState) -> ConversationTurnResult:
        active_menu = state.stage.value if state.available_options or state.pending_auth_field else None
        return ConversationTurnResult(
            text=state.response_text,
            final_outcome=state.final_outcome,
            active_menu=active_menu,
            menu_options=list(state.available_options),
            artifacts=dict(state.artifacts),
            adapter_results=dict(state.adapter_results),
            state_snapshot=state.model_dump(mode="json"),
        )

    def _append_history(self, state: ConversationSessionState) -> None:
        if state.latest_user_input:
            state.history.append(("human", state.latest_user_input))
        if state.response_text:
            state.history.append(("ai", state.response_text))

    def _default_prompt_for_stage(self, state: ConversationSessionState) -> str:
        prompts = {
            ConversationStage.ROOT: (
                "I can help with general information, account support, announcements, or "
                "feedback and contact options. What would you like help with?"
            ),
            ConversationStage.INFORMATION: (
                "I can help with store information, product information, or something else. "
                "Which would you like?"
            ),
            ConversationStage.REGISTRATION_TERMINAL: self._terminal_prompt_for_context(
                InteractionContext.REGISTRATION
            ),
            ConversationStage.LOGIN_TERMINAL: self._terminal_prompt_for_context(
                InteractionContext.LOGIN
            ),
            ConversationStage.FAIL_TERMINAL: (
                "I can still help with general communication options, send an SMS, or share "
                "general details. Which would you like?"
            ),
            ConversationStage.ANNOUNCEMENTS_CONTINUE: (
                f"{self.settings.announcements_message} If you'd like, I can also connect you "
                "to contact center support after that. Would you like me to continue?"
            ),
            ConversationStage.ANNOUNCEMENTS_TERMINAL: (
                "I can connect you to a human agent, or I can arrange a call back instead. "
                "Which works better for you?"
            ),
            ConversationStage.FEEDBACK_QUESTION: self.settings.feedback_question_prompt,
            ConversationStage.FEEDBACK_TERMINAL: (
                "I can connect you with a human agent, or I can submit a contact request for "
                "follow-up. Which would you prefer?"
            ),
        }
        return prompts.get(state.stage, "How can I help?")

    @staticmethod
    def _safe_response_text(
        preferred_text: str,
        fallback_text: str,
        *,
        required_fragments: tuple[str, ...] = (),
    ) -> str:
        normalized = preferred_text.strip()
        if not normalized:
            return fallback_text
        if required_fragments and not all(fragment in normalized for fragment in required_fragments):
            return fallback_text
        return normalized

    @staticmethod
    def _terminal_prompt_for_context(context: InteractionContext | None) -> str:
        if context == InteractionContext.REGISTRATION:
            return (
                "You're all set to continue. I can complete the registration, send the "
                "registration SMS confirmation, or send a general SMS with the next steps. "
                "Which would you prefer?"
            )
        return (
            "You're verified. I can continue with login, help with a balance update, or go "
            "over the account details. What would you like to do next?"
        )


def _text_to_tokens(text: str) -> list[str]:
    if not text:
        return []
    return re.findall(r"\S+\s*", text)
