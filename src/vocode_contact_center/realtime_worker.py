from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, AsyncGenerator, AsyncIterable, Awaitable, Callable, Protocol
from uuid import uuid4

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, status
from loguru import logger
from pydantic import BaseModel, Field
from vocode.streaming.models.audio import AudioEncoding, SamplingRate
from vocode.streaming.models.message import LLMToken
from vocode.streaming.models.synthesizer import ElevenLabsSynthesizerConfig
from vocode.streaming.synthesizer.eleven_labs_websocket_synthesizer import ElevenLabsWSSynthesizer
from vocode.streaming.utils import get_chunk_size_per_second

from vocode_contact_center.orchestration import ConversationOrchestrator
from vocode_contact_center.settings import ContactCenterSettings


class RealtimeSessionCreateRequest(BaseModel):
    call_context: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class RealtimeSessionCreateResponse(BaseModel):
    session_id: str
    transport: str
    websocket_path: str
    input_mode: str
    output_audio_encoding: str
    output_sample_rate: int
    legacy_telephony_available: bool


class StreamingLLM(Protocol):
    async def stream_response(
        self,
        *,
        session_id: str,
        call_context: str,
        committed_messages: list[tuple[str, str]],
        current_user_text: str,
        commit: bool = True,
    ) -> AsyncGenerator[str, None]:
        ...


class InputStreamingTTS(Protocol):
    @property
    def audio_encoding(self) -> str:
        ...

    @property
    def sample_rate(self) -> int:
        ...

    async def stream_tokens(self, tokens: AsyncIterable[str]) -> AsyncGenerator[bytes, None]:
        ...

    async def cancel_current_utterance(self) -> None:
        ...


class VoicebotStreamingLLM:
    def __init__(
        self,
        settings: ContactCenterSettings,
        *,
        conversation_orchestrator: ConversationOrchestrator | None = None,
    ):
        self.settings = settings
        if conversation_orchestrator is None:
            from vocode_contact_center.app import build_conversation_orchestrator

            conversation_orchestrator = build_conversation_orchestrator(settings)
        self.conversation_orchestrator = conversation_orchestrator

    async def stream_response(
        self,
        *,
        session_id: str,
        call_context: str,
        committed_messages: list[tuple[str, str]],
        current_user_text: str,
        commit: bool = True,
    ) -> AsyncGenerator[str, None]:
        result = (
            await self.conversation_orchestrator.run_turn(
                session_id,
                current_user_text,
                call_context=call_context,
                metadata={"transport": "realtime"},
            )
            if commit
            else await self.conversation_orchestrator.preview_turn(
                session_id,
                current_user_text,
                call_context=call_context,
                metadata={"transport": "realtime"},
            )
        )
        async for token in self.conversation_orchestrator.stream_text_response(result.text):
            yield token


class ElevenLabsInputStreamingTTS:
    def __init__(self, settings: ContactCenterSettings):
        if not settings.elevenlabs_use_websocket:
            raise ValueError(
                "Realtime voice requires ELEVENLABS_USE_WEBSOCKET=true so the TTS path "
                "can accept streaming input."
            )

        self._audio_encoding = AudioEncoding(settings.realtime_audio_encoding)
        self._sample_rate = SamplingRate(settings.realtime_sample_rate)
        self._chunk_size = max(
            get_chunk_size_per_second(self._audio_encoding, self._sample_rate) // 8,
            320,
        )
        self._config = ElevenLabsSynthesizerConfig(
            sampling_rate=self._sample_rate,
            audio_encoding=self._audio_encoding,
            api_key=settings.elevenlabs_api_key,
            voice_id=settings.elevenlabs_voice_id,
            model_id=settings.elevenlabs_model_id,
            optimize_streaming_latency=settings.elevenlabs_optimize_streaming_latency,
            experimental_websocket=True,
        )
        self._current_synth: ElevenLabsWSSynthesizer | None = None
        self._lock = asyncio.Lock()

    @property
    def audio_encoding(self) -> str:
        return self._audio_encoding.value

    @property
    def sample_rate(self) -> int:
        return int(self._sample_rate)

    async def stream_tokens(self, tokens: AsyncIterable[str]) -> AsyncGenerator[bytes, None]:
        synth = ElevenLabsWSSynthesizer(self._config)
        result_ready = asyncio.Event()
        result_holder: dict[str, Any] = {}

        async def sender() -> None:
            try:
                async for token in tokens:
                    if not token:
                        continue
                    await synth.send_token_to_synthesizer(
                        LLMToken(text=token),
                        chunk_size=self._chunk_size,
                    )
                    if "result" not in result_holder:
                        result_holder["result"] = synth.get_current_utterance_synthesis_result()
                        result_ready.set()
                if "result" in result_holder:
                    await synth.handle_end_of_turn()
            finally:
                result_ready.set()

        sender_task = asyncio.create_task(sender())
        async with self._lock:
            self._current_synth = synth
        try:
            await result_ready.wait()
            synthesis_result = result_holder.get("result")
            if synthesis_result is None:
                await sender_task
                return

            async for chunk_result in synthesis_result.chunk_generator:
                yield chunk_result.chunk

            await sender_task
        finally:
            sender_task.cancel()
            async with self._lock:
                if self._current_synth is synth:
                    self._current_synth = None
            await synth.tear_down()

    async def cancel_current_utterance(self) -> None:
        async with self._lock:
            synth = self._current_synth
        if synth is not None:
            await synth.cancel_websocket_tasks()


class RealtimeSnapshot(BaseModel):
    total_sessions_created: int
    active_sessions: int
    total_interruptions: int
    completed_responses: int


@dataclass
class RealtimeMetrics:
    total_sessions_created: int = 0
    total_interruptions: int = 0
    completed_responses: int = 0


@dataclass
class RealtimeVoiceSession:
    session_id: str
    call_context: str
    metadata: dict[str, str]
    send_event: Callable[[dict[str, Any]], Awaitable[None]]
    llm: StreamingLLM
    tts: InputStreamingTTS
    settings: ContactCenterSettings
    metrics: RealtimeMetrics
    committed_messages: list[tuple[str, str]] = field(default_factory=list)
    current_partial_text: str = ""
    current_final_text: str = ""
    _assistant_task: asyncio.Task | None = None
    _assistant_generation: int = 0
    _active_response_mode: str | None = None

    async def send_session_ready(self, *, legacy_telephony_available: bool) -> None:
        await self.send_event(
            {
                "type": "session_ready",
                "session_id": self.session_id,
                "transport": self.settings.realtime_transport,
                "input_mode": self.settings.realtime_input_mode,
                "output_audio_encoding": self.tts.audio_encoding,
                "output_sample_rate": self.tts.sample_rate,
                "legacy_telephony_available": legacy_telephony_available,
            }
        )

    async def handle_client_event(self, event: dict[str, Any]) -> None:
        event_type = str(event.get("type", "")).strip().lower()
        if event_type == "ping":
            await self.send_event({"type": "pong"})
            return
        if event_type == "barge_in":
            await self.interrupt(reason="barge_in")
            return
        if event_type == "user_partial":
            await self._handle_user_partial(str(event.get("text", "")))
            return
        if event_type == "user_final":
            await self._handle_user_final(str(event.get("text", "")))
            return
        await self.send_event(
            {
                "type": "error",
                "message": f"Unsupported realtime event type: {event_type or 'missing'}",
            }
        )

    async def _handle_user_partial(self, text: str) -> None:
        normalized = text.strip()
        if not normalized:
            return
        self.current_partial_text = normalized
        if self._assistant_task is not None:
            await self.interrupt(reason="user_speaking")
        if self._should_start_partial_response(normalized):
            await self._start_response(normalized, mode="partial")

    async def _handle_user_final(self, text: str) -> None:
        normalized = text.strip()
        if not normalized:
            return
        self.current_final_text = normalized
        self.current_partial_text = normalized
        await self.interrupt(reason="user_finalized")
        await self._start_response(normalized, mode="final")

    def _should_start_partial_response(self, text: str) -> bool:
        if not self.settings.realtime_allow_partial_responses:
            return False
        words = len(text.split())
        if words < self.settings.realtime_partial_response_min_words:
            return False
        if len(text) < self.settings.realtime_partial_response_min_chars:
            return False
        return True

    async def _start_response(self, source_text: str, *, mode: str) -> None:
        if self._assistant_task is not None and not self._assistant_task.done():
            return
        self._assistant_generation += 1
        generation = self._assistant_generation
        self._active_response_mode = mode
        self._assistant_task = asyncio.create_task(
            self._run_response(source_text=source_text, mode=mode, generation=generation)
        )

    async def _run_response(self, *, source_text: str, mode: str, generation: int) -> None:
        token_queue: asyncio.Queue[str | None] = asyncio.Queue()
        llm_started_at = perf_counter()
        first_text_sent = False
        first_audio_sent = False
        collected_tokens: list[str] = []

        async def token_source() -> AsyncGenerator[str, None]:
            while True:
                token = await token_queue.get()
                if token is None:
                    return
                yield token

        async def audio_worker() -> None:
            nonlocal first_audio_sent
            async for audio_chunk in self.tts.stream_tokens(token_source()):
                if generation != self._assistant_generation:
                    return
                if not first_audio_sent:
                    first_audio_sent = True
                    await self.send_event(
                        {
                            "type": "assistant_audio_started",
                            "latency_ms": round((perf_counter() - llm_started_at) * 1000, 1),
                            "response_mode": mode,
                        }
                    )
                await self.send_event(
                    {
                        "type": "assistant_audio",
                        "audio": base64.b64encode(audio_chunk).decode("ascii"),
                        "audio_encoding": self.tts.audio_encoding,
                        "sample_rate": self.tts.sample_rate,
                        "response_mode": mode,
                    }
                )

        audio_task = asyncio.create_task(audio_worker())
        try:
            await self.send_event({"type": "assistant_response_started", "response_mode": mode})
            async for token in self.llm.stream_response(
                session_id=self.session_id,
                call_context=self.call_context,
                committed_messages=self.committed_messages,
                current_user_text=source_text,
                commit=mode == "final",
            ):
                if generation != self._assistant_generation:
                    return
                collected_tokens.append(token)
                if not first_text_sent:
                    first_text_sent = True
                    await self.send_event(
                        {
                            "type": "assistant_text_started",
                            "latency_ms": round((perf_counter() - llm_started_at) * 1000, 1),
                            "response_mode": mode,
                        }
                    )
                await token_queue.put(token)
                await self.send_event(
                    {"type": "assistant_text", "text": token, "response_mode": mode}
                )
            await token_queue.put(None)
            await audio_task
            full_text = "".join(collected_tokens).strip()
            if mode == "final" and full_text:
                self.committed_messages.append(("human", source_text))
                self.committed_messages.append(("ai", full_text))
            self.metrics.completed_responses += 1
            await self.send_event(
                {
                    "type": "assistant_turn_end",
                    "response_mode": mode,
                    "text": full_text,
                }
            )
        except asyncio.CancelledError:
            await token_queue.put(None)
            await self.tts.cancel_current_utterance()
            audio_task.cancel()
            raise
        finally:
            if self._assistant_task is asyncio.current_task():
                self._assistant_task = None
            self._active_response_mode = None

    async def interrupt(self, *, reason: str) -> None:
        if self._assistant_task is None or self._assistant_task.done():
            return
        self.metrics.total_interruptions += 1
        self._assistant_generation += 1
        self._assistant_task.cancel()
        try:
            await self._assistant_task
        except asyncio.CancelledError:
            pass
        finally:
            self._assistant_task = None
        await self.send_event({"type": "assistant_interrupted", "reason": reason})

    async def close(self) -> None:
        await self.interrupt(reason="session_closed")


class RealtimeSessionManager:
    def __init__(
        self,
        settings: ContactCenterSettings,
        *,
        llm: StreamingLLM | None = None,
        tts_factory: Callable[[], InputStreamingTTS] | None = None,
        conversation_orchestrator: ConversationOrchestrator | None = None,
        legacy_telephony_available: bool = False,
    ):
        self.settings = settings
        if conversation_orchestrator is None:
            from vocode_contact_center.app import build_conversation_orchestrator

            conversation_orchestrator = build_conversation_orchestrator(settings)
        self.conversation_orchestrator = conversation_orchestrator
        self.llm = llm or VoicebotStreamingLLM(
            settings,
            conversation_orchestrator=self.conversation_orchestrator,
        )
        self.tts_factory = tts_factory or (lambda: ElevenLabsInputStreamingTTS(settings))
        self.legacy_telephony_available = legacy_telephony_available
        self.metrics = RealtimeMetrics()
        self._sessions: dict[str, RealtimeVoiceSession] = {}

    def create_session(self, request: RealtimeSessionCreateRequest) -> RealtimeSessionCreateResponse:
        session_id = uuid4().hex
        self.metrics.total_sessions_created += 1
        self._sessions[session_id] = RealtimeVoiceSession(
            session_id=session_id,
            call_context=request.call_context or "Realtime session metadata is unavailable.",
            metadata=request.metadata,
            send_event=self._missing_sender,
            llm=self.llm,
            tts=self.tts_factory(),
            settings=self.settings,
            metrics=self.metrics,
        )
        return RealtimeSessionCreateResponse(
            session_id=session_id,
            transport=self.settings.realtime_transport,
            websocket_path=f"/realtime/sessions/{session_id}/ws",
            input_mode=self.settings.realtime_input_mode,
            output_audio_encoding=self.settings.realtime_audio_encoding,
            output_sample_rate=self.settings.realtime_sample_rate,
            legacy_telephony_available=self.legacy_telephony_available,
        )

    async def connect(self, session_id: str, websocket: WebSocket) -> None:
        session = self._sessions.get(session_id)
        if session is None:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="unknown_session")
            return

        await websocket.accept()

        async def send_event(payload: dict[str, Any]) -> None:
            await websocket.send_json(payload)

        session.send_event = send_event
        await session.send_session_ready(
            legacy_telephony_available=self.legacy_telephony_available
        )
        try:
            while True:
                payload = await websocket.receive_json()
                await session.handle_client_event(payload)
        except WebSocketDisconnect:
            logger.info("Realtime websocket disconnected for session {}", session_id)
        finally:
            await session.close()
            self._sessions.pop(session_id, None)

    def snapshot(self) -> RealtimeSnapshot:
        return RealtimeSnapshot(
            total_sessions_created=self.metrics.total_sessions_created,
            active_sessions=len(self._sessions),
            total_interruptions=self.metrics.total_interruptions,
            completed_responses=self.metrics.completed_responses,
        )

    @staticmethod
    async def _missing_sender(payload: dict[str, Any]) -> None:
        raise RuntimeError(f"Realtime session is not connected. Dropped payload: {payload!r}")


def create_realtime_router(
    settings: ContactCenterSettings,
    *,
    manager: RealtimeSessionManager,
    realtime_ready: bool,
) -> APIRouter:
    router = APIRouter(prefix="/realtime", tags=["realtime"])

    @router.post("/sessions", response_model=RealtimeSessionCreateResponse)
    async def create_session(request: RealtimeSessionCreateRequest):
        if not realtime_ready:
            raise HTTPException(
                status_code=503,
                detail={
                    "message": "Realtime voice is not configured.",
                    "missing_runtime_values": settings.missing_realtime_values(),
                },
            )
        return manager.create_session(request)

    @router.websocket("/sessions/{session_id}/ws")
    async def connect_session(websocket: WebSocket, session_id: str):
        if not realtime_ready:
            await websocket.close(
                code=status.WS_1008_POLICY_VIOLATION,
                reason="realtime_not_configured",
            )
            return
        await manager.connect(session_id, websocket)

    return router
