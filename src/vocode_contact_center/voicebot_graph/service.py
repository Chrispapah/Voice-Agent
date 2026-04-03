from __future__ import annotations

import re
from dataclasses import dataclass
from typing import AsyncGenerator

from loguru import logger

from vocode_contact_center.settings import ContactCenterSettings
from vocode_contact_center.voicebot_graph.adapters.base import (
    AuthenticationAdapter,
    GenesysAdapter,
    SmsSender,
)
from vocode_contact_center.voicebot_graph.adapters.stub import (
    StubAuthenticationAdapter,
    StubGenesysAdapter,
    StubSmsSender,
)
from vocode_contact_center.voicebot_graph.graph import build_voicebot_graph
from vocode_contact_center.voicebot_graph.state import (
    VoicebotGraphState,
    clear_turn_fields,
    clone_state,
    initial_graph_state,
    reset_path_state,
)


@dataclass
class VoicebotTurnResult:
    text: str
    final_outcome: str | None
    active_menu: str | None
    menu_options: list[str]
    artifacts: dict[str, str]
    adapter_results: dict[str, object]
    state_snapshot: VoicebotGraphState


class VoicebotGraphService:
    def __init__(
        self,
        settings: ContactCenterSettings,
        *,
        auth_adapter: AuthenticationAdapter | None = None,
        genesys_adapter: GenesysAdapter | None = None,
        sms_sender: SmsSender | None = None,
    ) -> None:
        self.settings = settings
        self.auth_adapter = auth_adapter or StubAuthenticationAdapter(settings)
        self.genesys_adapter = genesys_adapter or StubGenesysAdapter()
        self.sms_sender = sms_sender or StubSmsSender()
        self._graph = build_voicebot_graph(
            settings=settings,
            auth_adapter=self.auth_adapter,
            genesys_adapter=self.genesys_adapter,
            sms_sender=self.sms_sender,
        )
        self._sessions: dict[str, VoicebotGraphState] = {}

    def clear_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    async def run_turn(
        self,
        session_id: str,
        user_text: str,
        *,
        call_context: str,
        metadata: dict[str, str] | None = None,
        commit: bool = True,
    ) -> VoicebotTurnResult:
        state = self._prepare_state(
            session_id,
            user_text,
            call_context=call_context,
            metadata=metadata,
        )
        logger.info(
            "Graph run_turn session={} commit={} input={!r} current_path={} active_menu={} pending_auth_field={} conversation_memory_keys={}",
            session_id,
            commit,
            user_text,
            state.get("current_path"),
            state.get("active_menu"),
            state.get("pending_auth_field"),
            sorted(state.get("conversation_memory", {}).keys()),
        )
        result_state = await self._graph.ainvoke(state)
        logger.info(
            "Graph run_turn result session={} final_outcome={} active_menu={} next_route={} pending_auth_field={} conversation_memory_keys={}",
            session_id,
            result_state.get("final_outcome"),
            result_state.get("active_menu"),
            result_state.get("route_decision"),
            result_state.get("pending_auth_field"),
            sorted(result_state.get("conversation_memory", {}).keys()),
        )
        if commit:
            self._sessions[session_id] = clone_state(result_state)
        return VoicebotTurnResult(
            text=result_state.get("response_text", ""),
            final_outcome=result_state.get("final_outcome"),
            active_menu=result_state.get("active_menu"),
            menu_options=list(result_state.get("menu_options", [])),
            artifacts=dict(result_state.get("artifacts", {})),
            adapter_results=dict(result_state.get("adapter_results", {})),
            state_snapshot=clone_state(result_state),
        )

    async def preview_turn(
        self,
        session_id: str,
        user_text: str,
        *,
        call_context: str,
        metadata: dict[str, str] | None = None,
    ) -> VoicebotTurnResult:
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

    def _prepare_state(
        self,
        session_id: str,
        user_text: str,
        *,
        call_context: str,
        metadata: dict[str, str] | None,
    ) -> VoicebotGraphState:
        existing = self._sessions.get(session_id)
        if existing is None:
            state = initial_graph_state(
                session_id=session_id,
                call_context=call_context,
                metadata=metadata,
            )
        else:
            state = clone_state(existing)

        if state.get("final_outcome") and not state.get("active_menu"):
            state = reset_path_state(state)
            state["session_id"] = session_id
            state["call_context"] = call_context
            state["metadata"] = dict(metadata or {})
        else:
            state = clear_turn_fields(state)
            state["call_context"] = call_context
            if metadata:
                merged = dict(state.get("metadata", {}))
                merged.update(metadata)
                state["metadata"] = merged

        state["latest_user_input"] = user_text.strip()
        return state


def _text_to_tokens(text: str) -> list[str]:
    if not text:
        return []
    return re.findall(r"\S+\s*", text)
