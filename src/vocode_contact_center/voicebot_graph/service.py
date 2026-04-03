from __future__ import annotations

import asyncio
import re
import threading
from dataclasses import dataclass
from typing import AsyncGenerator

from loguru import logger

from vocode_contact_center.settings import ContactCenterSettings
from vocode_contact_center.voicebot_graph.adapters.base import (
    AuthenticationAdapter,
    GenesysAdapter,
    SmsRequest,
    SmsResult,
    SmsSender,
)
from vocode_contact_center.voicebot_graph.adapters.stub import (
    StubAuthenticationAdapter,
    StubGenesysAdapter,
    StubSmsSender,
)
from vocode_contact_center.voicebot_graph.adapters.twilio_sms import TwilioSmsSender
from vocode_contact_center.voicebot_graph.graph import build_voicebot_graph
from vocode_contact_center.voicebot_graph.spoken import spoken_preview_from_state
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


def _text_to_tokens(text: str) -> list[str]:
    if not text:
        return []
    return re.findall(r"\S+\s*", text)


def _spoken_delta(last_emitted: str, new_full: str) -> tuple[str, str]:
    if new_full.startswith(last_emitted):
        return new_full, new_full[len(last_emitted) :]
    return new_full, new_full


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
        self._session_lock = threading.Lock()
        schedule_sms = None
        if settings.defer_sms_send_in_background and isinstance(self.sms_sender, TwilioSmsSender):
            schedule_sms = self._enqueue_background_sms
        self._graph = build_voicebot_graph(
            settings=settings,
            auth_adapter=self.auth_adapter,
            genesys_adapter=self.genesys_adapter,
            sms_sender=self.sms_sender,
            schedule_background_sms=schedule_sms,
        )
        self._sessions: dict[str, VoicebotGraphState] = {}

    def clear_session(self, session_id: str) -> None:
        with self._session_lock:
            self._sessions.pop(session_id, None)

    def _enqueue_background_sms(self, request: SmsRequest) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.error("Cannot defer SMS: no running event loop session={}", request.session_id)
            return
        loop.create_task(self._complete_background_sms(request))

    async def _complete_background_sms(self, request: SmsRequest) -> None:
        sid = request.session_id
        try:
            result = await self.sms_sender.send(request)
        except Exception as exc:
            logger.exception("Deferred SMS send failed session={}", sid)
            result = SmsResult(
                status="failed",
                error_message=str(exc),
                metadata={"provider": "exception"},
            )
        with self._session_lock:
            existing = self._sessions.get(sid)
            if not existing:
                return
            merged = clone_state(existing)
            ar = dict(merged.get("adapter_results") or {})
            sms_meta: dict[str, object] = {
                "status": result.status,
                "metadata": {
                    **result.metadata,
                    **(
                        {"provider_message_id": result.provider_message_id}
                        if result.provider_message_id
                        else {}
                    ),
                    **({"error_message": result.error_message} if result.error_message else {}),
                },
            }
            ar["sms"] = sms_meta
            merged["adapter_results"] = ar
            artifacts = dict(merged.get("artifacts") or {})
            if result.status == "sent":
                cd = dict(merged.get("collected_data") or {})
                cd["sms_confirmed"] = "true"
                merged["collected_data"] = cd
                artifacts["sms_status"] = "sent"
                if result.provider_message_id:
                    artifacts["sms_message_id"] = result.provider_message_id
            else:
                artifacts["sms_status"] = "failed"
            merged["artifacts"] = artifacts
            self._sessions[sid] = merged

    def export_session(self, session_id: str) -> VoicebotGraphState | None:
        with self._session_lock:
            snap = self._sessions.get(session_id)
            return clone_state(snap) if snap is not None else None

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
            with self._session_lock:
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
            session_id,
            user_text,
            call_context=call_context,
            metadata=metadata,
        )
        logger.info(
            "Graph stream_generate_response session={} commit={} input={!r} current_path={} active_menu={} pending_auth_field={}",
            session_id,
            commit,
            user_text,
            state.get("current_path"),
            state.get("active_menu"),
            state.get("pending_auth_field"),
        )
        last_spoken = ""
        final_state: VoicebotGraphState | None = None
        async for values in self._graph.astream(state, stream_mode="values"):
            final_state = values
            spoken = spoken_preview_from_state(values)
            last_spoken, delta = _spoken_delta(last_spoken, spoken)
            for tok in _text_to_tokens(delta):
                yield tok
        if final_state is None:
            logger.warning("Graph stream ended with no state session={}", session_id)
            return
        logger.info(
            "Graph stream_generate_response done session={} final_outcome={} active_menu={}",
            session_id,
            final_state.get("final_outcome"),
            final_state.get("active_menu"),
        )
        if commit:
            with self._session_lock:
                self._sessions[session_id] = clone_state(final_state)

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
        with self._session_lock:
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
