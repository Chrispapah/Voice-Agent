from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from loguru import logger
from vocode.streaming.agent.base_agent import AgentResponseMessage, RespondAgent
from vocode.streaming.models.actions import EndOfTurn
from vocode.streaming.models.agent import AgentConfig
from vocode.streaming.models.message import BaseMessage, LLMToken
from vocode.streaming.models.transcriber import Transcription

from ai_sdr_agent.graph.service import SDRConversationService
from ai_sdr_agent.services.latency_analytics import (
    clear_phone_turn_on_error,
    mark_phone_turn_graph_done,
    mark_phone_turn_respond_enter,
)

_DUPLICATE_FINAL_WINDOW_S = 0.75
_DUPLICATE_AFTER_INTERRUPT_WINDOW_S = 2.5
_STREAM_PUNCTUATION = ".?,!;:"
_STREAM_HARD_FLUSH_CHARS = 24


def _format_details(**details: object) -> str:
    parts: list[str] = []
    for key, value in details.items():
        if value is None:
            continue
        if isinstance(value, str):
            parts.append(f"{key}={value!r}")
        else:
            parts.append(f"{key}={value}")
    return " ".join(parts)


def _log_agent_step_latency(
    conversation_id: str,
    step_name: str,
    latency_ms: float,
    **details: object,
) -> None:
    detail_text = _format_details(**details)
    if detail_text:
        logger.info(
            "Agent step latency conversation_id={} step={} latency_ms={:.0f} details={}",
            conversation_id,
            step_name,
            latency_ms,
            detail_text,
        )
        return
    logger.info(
        "Agent step latency conversation_id={} step={} latency_ms={:.0f}",
        conversation_id,
        step_name,
        latency_ms,
    )


def _normalize_interrupt_text(text: str) -> str:
    return " ".join(text.split()).strip().lower()


def _log_interrupt_buffer_event(
    conversation_id: str,
    decision: str,
    **details: object,
) -> None:
    detail_text = _format_details(**details)
    if detail_text:
        logger.info(
            "Interrupt buffer conversation_id={} decision={} details={}",
            conversation_id,
            decision,
            detail_text,
        )
        return
    logger.info(
        "Interrupt buffer conversation_id={} decision={}",
        conversation_id,
        decision,
    )


@dataclass
class _TurnOutcome:
    response: str | None
    should_stop: bool
    skip_reason: str | None = None


@dataclass
class _LiveResponseContext:
    emit_chunk: Callable[[str], Awaitable[None]]
    active: bool = True
    used: bool = False
    emitted_any: bool = False
    buffer: str = ""


@dataclass
class _TurnRequest:
    text: str
    normalized: str
    is_interrupt: bool
    sequence: int | None
    generation: int
    future: asyncio.Future[_TurnOutcome]
    live_response: _LiveResponseContext | None = None
    started_at: float = field(default_factory=time.perf_counter)


@dataclass
class _ConversationRuntimeState:
    turn_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    latest_interrupt_sequence: int = 0
    next_generation: int = 0
    current_turn: _TurnRequest | None = None
    pending_turn: _TurnRequest | None = None
    in_flight_turn_task: asyncio.Task[None] | None = None
    last_committed_text: str | None = None
    last_committed_normalized: str | None = None
    last_committed_source: str | None = None
    last_committed_at: float = 0.0


class SDRAgentConfig(AgentConfig, type="agent_sdr"):  # type: ignore[misc]
    lead_id: str
    calendar_id: str
    sales_rep_name: str
    initial_message_text: str


class SDRVocodeAgent(RespondAgent[SDRAgentConfig]):
    def __init__(
        self,
        agent_config: SDRAgentConfig,
        *,
        conversation_service: SDRConversationService,
        **kwargs,
    ):
        super().__init__(agent_config=agent_config, **kwargs)
        self.conversation_service = conversation_service
        self._initialized_conversations: set[str] = set()
        self._conversation_runtime: dict[str, _ConversationRuntimeState] = {}

    def _get_runtime_state(self, conversation_id: str) -> _ConversationRuntimeState:
        return self._conversation_runtime.setdefault(conversation_id, _ConversationRuntimeState())

    @staticmethod
    def _resolve_turn_future(
        request: _TurnRequest,
        *,
        response: str | None,
        should_stop: bool,
        skip_reason: str | None = None,
    ) -> None:
        if not request.future.done():
            request.future.set_result(
                _TurnOutcome(
                    response=response,
                    should_stop=should_stop,
                    skip_reason=skip_reason,
                )
            )

    def _register_interrupt(
        self,
        conversation_id: str,
        human_input: str,
        runtime_state: _ConversationRuntimeState,
    ) -> tuple[int | None, str, str | None]:
        normalized_input = _normalize_interrupt_text(human_input)
        if not normalized_input:
            _log_interrupt_buffer_event(
                conversation_id,
                "drop_empty_interrupt",
            )
            return None, normalized_input, "empty_interrupt"
        runtime_state.latest_interrupt_sequence += 1
        return runtime_state.latest_interrupt_sequence, normalized_input, None

    def _recent_duplicate_reason(
        self,
        conversation_id: str,
        human_input: str,
        normalized_input: str,
        runtime_state: _ConversationRuntimeState,
        *,
        is_interrupt: bool,
    ) -> str | None:
        if is_interrupt or not normalized_input:
            return None
        if runtime_state.last_committed_normalized != normalized_input:
            return None
        age_s = time.perf_counter() - runtime_state.last_committed_at
        previous_text = runtime_state.last_committed_text or runtime_state.last_committed_normalized
        if runtime_state.last_committed_source == "interrupt":
            if age_s <= _DUPLICATE_AFTER_INTERRUPT_WINDOW_S:
                _log_interrupt_buffer_event(
                    conversation_id,
                    "drop_duplicate_final_after_interrupt",
                    previous_text=previous_text,
                    text=human_input,
                    age_ms=f"{age_s * 1000:.0f}",
                )
                return "duplicate_final_after_interrupt"
            return None
        if runtime_state.last_committed_source == "final" and age_s <= _DUPLICATE_FINAL_WINDOW_S:
            _log_interrupt_buffer_event(
                conversation_id,
                "drop_duplicate_final",
                previous_text=previous_text,
                text=human_input,
                age_ms=f"{age_s * 1000:.0f}",
            )
            return "duplicate_final"
        return None

    @staticmethod
    def _record_committed_input(
        runtime_state: _ConversationRuntimeState,
        *,
        human_input: str,
        normalized_input: str,
        is_interrupt: bool,
    ) -> None:
        if not normalized_input:
            return
        runtime_state.last_committed_text = human_input
        runtime_state.last_committed_normalized = normalized_input
        runtime_state.last_committed_source = "interrupt" if is_interrupt else "final"
        runtime_state.last_committed_at = time.perf_counter()

    @staticmethod
    async def _emit_live_chunk(request: _TurnRequest, chunk: str) -> None:
        live_response = request.live_response
        if live_response is None or not live_response.active or not chunk:
            return
        live_response.buffer += chunk
        while live_response.buffer:
            boundary_index = next(
                (idx for idx, char in enumerate(live_response.buffer) if char in _STREAM_PUNCTUATION),
                -1,
            )
            if boundary_index >= 2:
                output = live_response.buffer[: boundary_index + 1].strip()
                live_response.buffer = live_response.buffer[boundary_index + 1 :].lstrip()
                if output:
                    live_response.used = True
                    live_response.emitted_any = True
                    await live_response.emit_chunk(output if output.endswith(" ") else output + " ")
                continue
            if len(live_response.buffer) >= _STREAM_HARD_FLUSH_CHARS:
                space_index = live_response.buffer.rfind(" ")
                if space_index > 0:
                    output = live_response.buffer[:space_index].strip()
                    live_response.buffer = live_response.buffer[space_index + 1 :].lstrip()
                    if output:
                        live_response.used = True
                        live_response.emitted_any = True
                        await live_response.emit_chunk(
                            output if output.endswith(" ") else output + " "
                        )
                    continue
            break

    @staticmethod
    async def _flush_live_response(request: _TurnRequest) -> None:
        live_response = request.live_response
        if live_response is None or not live_response.active:
            return
        if not live_response.buffer:
            return
        buffered = live_response.buffer
        live_response.buffer = ""
        live_response.used = True
        live_response.emitted_any = True
        await live_response.emit_chunk(buffered if buffered.endswith(" ") else buffered + " ")

    async def _ensure_session_initialized(self, conversation_id: str) -> None:
        if conversation_id in self._initialized_conversations:
            return
        start_session_t0 = time.perf_counter()
        await self.conversation_service.start_session(
            self.agent_config.lead_id,
            conversation_id=conversation_id,
        )
        start_session_ms = (time.perf_counter() - start_session_t0) * 1000
        _log_agent_step_latency(
            conversation_id,
            "start_session",
            start_session_ms,
            lead_id=self.agent_config.lead_id,
        )
        self._initialized_conversations.add(conversation_id)
        logger.info(
            "Initialized SDR session from phone call conversation_id={} lead_id={}",
            conversation_id,
            self.agent_config.lead_id,
        )

    @staticmethod
    def _build_turn_request(
        runtime_state: _ConversationRuntimeState,
        *,
        human_input: str,
        normalized_input: str,
        is_interrupt: bool,
        sequence: int | None,
        live_response: _LiveResponseContext | None = None,
    ) -> _TurnRequest:
        loop = asyncio.get_running_loop()
        runtime_state.next_generation += 1
        return _TurnRequest(
            text=human_input,
            normalized=normalized_input,
            is_interrupt=is_interrupt,
            sequence=sequence,
            generation=runtime_state.next_generation,
            future=loop.create_future(),
            live_response=live_response,
        )

    async def _start_turn_request(
        self,
        conversation_id: str,
        runtime_state: _ConversationRuntimeState,
        *,
        request: _TurnRequest,
    ) -> None:
        await self._ensure_session_initialized(conversation_id)
        runtime_state.current_turn = request
        runtime_state.in_flight_turn_task = asyncio.create_task(
            self._run_turn_request(conversation_id, request)
        )
        _log_interrupt_buffer_event(
            conversation_id,
            "start_inflight_turn",
            generation=request.generation,
            text=request.text,
            source="interrupt" if request.is_interrupt else "final",
        )

    async def _run_turn_request(
        self,
        conversation_id: str,
        request: _TurnRequest,
    ) -> None:
        runtime_state = self._get_runtime_state(conversation_id)
        handle_turn_t0 = time.perf_counter()
        try:
            if request.live_response is not None:
                streamed_turn = await self.conversation_service.start_streamed_turn(
                    conversation_id,
                    request.text,
                )
                async for chunk in streamed_turn.chunks:
                    await self._emit_live_chunk(request, chunk)
                await self._flush_live_response(request)
                state = await streamed_turn.final_state_task
            else:
                state = await self.conversation_service.handle_turn(conversation_id, request.text)
            handle_turn_ms = (time.perf_counter() - handle_turn_t0) * 1000
        except Exception:
            _log_agent_step_latency(
                conversation_id,
                "respond_failed",
                (time.perf_counter() - request.started_at) * 1000,
            )
            logger.exception(
                "SDR agent background turn failed conversation_id={} text={!r}",
                conversation_id,
                request.text,
            )
            async with runtime_state.turn_lock:
                if runtime_state.current_turn is request:
                    runtime_state.current_turn = None
                    runtime_state.in_flight_turn_task = None
                    next_request = runtime_state.pending_turn
                    runtime_state.pending_turn = None
                    if next_request is not None:
                        await self._start_turn_request(
                            conversation_id,
                            runtime_state,
                            request=next_request,
                        )
                else:
                    next_request = None
            self._resolve_turn_future(
                request,
                response="I'm sorry, I'm having a technical issue. Could you hold on a moment?",
                should_stop=False,
            )
            return
        mark_phone_turn_graph_done(conversation_id)
        async with runtime_state.turn_lock:
            if runtime_state.current_turn is not request:
                if request.live_response is not None:
                    request.live_response.active = False
                self._resolve_turn_future(
                    request,
                    response=None,
                    should_stop=False,
                    skip_reason="stale_completed_reply",
                )
                return
            _log_agent_step_latency(
                conversation_id,
                "handle_turn",
                handle_turn_ms,
                turn_count=state["turn_count"],
                route_decision=state["route_decision"],
            )
            self._record_committed_input(
                runtime_state,
                human_input=request.text,
                normalized_input=request.normalized,
                is_interrupt=request.is_interrupt,
            )
            response = state["last_agent_response"]
            queued_request = runtime_state.pending_turn
            next_request = queued_request if response else None
            abandoned_request = queued_request if not response else None
            suppress_for_pending = next_request is not None
            suppress_for_stream = (
                request.live_response is not None
                and request.live_response.used
            )
            runtime_state.current_turn = None
            runtime_state.pending_turn = None
            runtime_state.in_flight_turn_task = None
            if request.live_response is not None:
                request.live_response.active = False
            if next_request is not None:
                await self._start_turn_request(
                    conversation_id,
                    runtime_state,
                    request=next_request,
                )
        if abandoned_request is not None:
            self._resolve_turn_future(
                abandoned_request,
                response=None,
                should_stop=False,
                skip_reason="conversation_complete",
            )
        if not response:
            logger.info("Conversation complete, no reply conversation_id={}", conversation_id)
            clear_phone_turn_on_error(conversation_id)
            self._resolve_turn_future(
                request,
                response=None,
                should_stop=True,
            )
            return
        if suppress_for_pending:
            self._resolve_turn_future(
                request,
                response=None,
                should_stop=False,
                skip_reason="stale_completed_reply",
            )
            return
        if suppress_for_stream:
            self._resolve_turn_future(
                request,
                response=None,
                should_stop=False,
                skip_reason="streamed_reply_delivered",
            )
            return
        logger.info(
            "SDR agent generated reply conversation_id={} text={!r}",
            conversation_id,
            response,
        )
        self._resolve_turn_future(
            request,
            response=response,
            should_stop=False,
        )

    async def _queue_turn_request(
        self,
        conversation_id: str,
        runtime_state: _ConversationRuntimeState,
        *,
        request: _TurnRequest,
    ) -> None:
        existing_request = runtime_state.current_turn
        if existing_request is None:
            if request.is_interrupt:
                _log_interrupt_buffer_event(
                    conversation_id,
                    "flush_interrupt",
                    sequence=request.sequence,
                    text=request.text,
                )
            await self._start_turn_request(
                conversation_id,
                runtime_state,
                request=request,
            )
            task = runtime_state.in_flight_turn_task
            assert task is not None
            task.add_done_callback(lambda completed_task: completed_task.exception())
            return
        if existing_request.normalized == request.normalized:
            _log_interrupt_buffer_event(
                conversation_id,
                "drop_current_repeat",
                text=request.text,
                active_text=existing_request.text,
            )
            self._resolve_turn_future(
                request,
                response=None,
                should_stop=False,
                skip_reason="duplicate_inflight_turn",
            )
            return
        pending_request = runtime_state.pending_turn
        if pending_request is not None:
            _log_interrupt_buffer_event(
                conversation_id,
                "replace_pending_turn",
                previous_text=pending_request.text,
                text=request.text,
            )
            self._resolve_turn_future(
                pending_request,
                response=None,
                should_stop=False,
                skip_reason="superseded_pending_turn",
            )
        else:
            _log_interrupt_buffer_event(
                conversation_id,
                "buffer_pending_turn",
                text=request.text,
                source="interrupt" if request.is_interrupt else "final",
            )
        runtime_state.pending_turn = request
        if request.is_interrupt:
            _log_interrupt_buffer_event(
                conversation_id,
                "queue_interrupt_refinement",
                sequence=request.sequence,
                text=request.text,
            )

    async def _enqueue_turn(
        self,
        *,
        human_input: str,
        conversation_id: str,
        is_interrupt: bool,
        live_response: _LiveResponseContext | None = None,
    ) -> tuple[_TurnRequest | None, float]:
        respond_start = time.perf_counter()
        runtime_state = self._get_runtime_state(conversation_id)
        interrupt_sequence: int | None = None
        normalized_input = _normalize_interrupt_text(human_input)
        logger.info(
            "SDR agent received speech conversation_id={} is_interrupt={} text={!r}",
            conversation_id,
            is_interrupt,
            human_input,
        )
        if is_interrupt:
            interrupt_sequence, normalized_input, skip_reason = self._register_interrupt(
                conversation_id,
                human_input,
                runtime_state,
            )
            if skip_reason is not None:
                total_ms = (time.perf_counter() - respond_start) * 1000
                _log_agent_step_latency(
                    conversation_id,
                    "respond_coalesced_skip",
                    total_ms,
                    reason=skip_reason,
                )
                return None, respond_start
        async with runtime_state.turn_lock:
            if is_interrupt and interrupt_sequence != runtime_state.latest_interrupt_sequence:
                _log_interrupt_buffer_event(
                    conversation_id,
                    "skip_stale_interrupt",
                    sequence=interrupt_sequence,
                    latest_sequence=runtime_state.latest_interrupt_sequence,
                    text=human_input,
                )
                total_ms = (time.perf_counter() - respond_start) * 1000
                _log_agent_step_latency(
                    conversation_id,
                    "respond_coalesced_skip",
                    total_ms,
                    reason="stale_interrupt",
                    sequence=interrupt_sequence,
                    latest_sequence=runtime_state.latest_interrupt_sequence,
                )
                return None, respond_start
            queue_wait_ms = (time.perf_counter() - respond_start) * 1000
            _log_agent_step_latency(
                conversation_id,
                "respond_queue_wait",
                queue_wait_ms,
                is_interrupt=is_interrupt,
            )
            duplicate_reason = self._recent_duplicate_reason(
                conversation_id,
                human_input,
                normalized_input,
                runtime_state,
                is_interrupt=is_interrupt,
            )
            if duplicate_reason is not None:
                total_ms = (time.perf_counter() - respond_start) * 1000
                _log_agent_step_latency(
                    conversation_id,
                    "respond_coalesced_skip",
                    total_ms,
                    reason=duplicate_reason,
                )
                return None, respond_start
            request = self._build_turn_request(
                runtime_state,
                human_input=human_input,
                normalized_input=normalized_input,
                is_interrupt=is_interrupt,
                sequence=interrupt_sequence,
                live_response=live_response,
            )
            await self._queue_turn_request(
                conversation_id,
                runtime_state,
                request=request,
            )
        mark_phone_turn_respond_enter(conversation_id)
        return request, respond_start

    def start(self) -> None:
        super().start()
        logger.info(
            "SDRVocodeAgent started lead_id={} generate_responses={}",
            self.agent_config.lead_id,
            self.agent_config.generate_responses,
        )
        opening = (self.agent_config.initial_message_text or "").strip()
        if opening:
            self.produce_interruptible_agent_response_event_nonblocking(
                AgentResponseMessage(
                    message=BaseMessage(text=opening),
                    is_first=True,
                ),
                is_interruptible=self.agent_config.allow_agent_to_be_cut_off,
            )
        self.produce_interruptible_agent_response_event_nonblocking(
            AgentResponseMessage(message=EndOfTurn()),
            is_interruptible=self.agent_config.allow_agent_to_be_cut_off,
        )

    async def handle_respond(self, transcription: Transcription, conversation_id: str) -> bool:
        if not self.conversation_service.dependencies.brain.supports_response_token_stream():
            return await super().handle_respond(transcription, conversation_id)

        is_first_chunk = True

        async def _emit_chunk(text: str) -> None:
            nonlocal is_first_chunk
            if not text:
                return
            using_input_streaming = bool(
                hasattr(self, "conversation_state_manager")
                and self.conversation_state_manager.using_input_streaming_synthesizer()
            )
            message_cls = LLMToken if using_input_streaming else BaseMessage
            self.produce_interruptible_agent_response_event_nonblocking(
                AgentResponseMessage(
                    message=message_cls(text=text),
                    is_first=is_first_chunk,
                ),
                is_interruptible=self.agent_config.allow_agent_to_be_cut_off,
            )
            is_first_chunk = False

        live_response = _LiveResponseContext(emit_chunk=_emit_chunk)
        request, respond_start = await self._enqueue_turn(
            human_input=transcription.message,
            conversation_id=conversation_id,
            is_interrupt=transcription.is_interrupt,
            live_response=live_response,
        )
        if request is None:
            logger.debug("No streamed response generated")
            return False
        try:
            outcome = await asyncio.shield(request.future)
        except asyncio.CancelledError:
            live_response.active = False
            total_ms = (time.perf_counter() - respond_start) * 1000
            _log_agent_step_latency(
                conversation_id,
                "respond_wait_cancelled",
                total_ms,
                is_interrupt=transcription.is_interrupt,
                generation=request.generation,
            )
            raise
        if outcome.skip_reason is not None and outcome.skip_reason != "streamed_reply_delivered":
            total_ms = (time.perf_counter() - respond_start) * 1000
            _log_agent_step_latency(
                conversation_id,
                "respond_coalesced_skip",
                total_ms,
                reason=outcome.skip_reason,
                generation=request.generation,
            )
            return False
        if not live_response.emitted_any:
            if outcome.response:
                await _emit_chunk(outcome.response)
            else:
                total_ms = (time.perf_counter() - respond_start) * 1000
                _log_agent_step_latency(
                    conversation_id,
                    "respond_total",
                    total_ms,
                    generation=request.generation,
                    streamed=True,
                )
                return outcome.should_stop
        self.produce_interruptible_agent_response_event_nonblocking(
            AgentResponseMessage(message=EndOfTurn()),
            is_interruptible=self.agent_config.allow_agent_to_be_cut_off,
        )
        total_ms = (time.perf_counter() - respond_start) * 1000
        _log_agent_step_latency(
            conversation_id,
            "respond_total",
            total_ms,
            generation=request.generation,
            streamed=True,
        )
        return outcome.should_stop

    async def respond(
        self,
        human_input: str,
        conversation_id: str,
        is_interrupt: bool = False,
    ) -> tuple[str | None, bool]:
        request, respond_start = await self._enqueue_turn(
            human_input=human_input,
            conversation_id=conversation_id,
            is_interrupt=is_interrupt,
        )
        if request is None:
            return None, False
        try:
            outcome = await asyncio.shield(request.future)
        except asyncio.CancelledError:
            total_ms = (time.perf_counter() - respond_start) * 1000
            _log_agent_step_latency(
                conversation_id,
                "respond_wait_cancelled",
                total_ms,
                is_interrupt=is_interrupt,
                generation=request.generation,
            )
            raise
        if outcome.skip_reason is not None:
            total_ms = (time.perf_counter() - respond_start) * 1000
            _log_agent_step_latency(
                conversation_id,
                "respond_coalesced_skip",
                total_ms,
                reason=outcome.skip_reason,
                generation=request.generation,
            )
            return None, False
        total_ms = (time.perf_counter() - respond_start) * 1000
        _log_agent_step_latency(
            conversation_id,
            "respond_total",
            total_ms,
            generation=request.generation,
        )
        return outcome.response, outcome.should_stop


def build_agent_config(
    *,
    lead_id: str,
    calendar_id: str,
    sales_rep_name: str,
    initial_message_text: str,
    allow_agent_to_be_cut_off: bool = True,
    interrupt_sensitivity: str = "high",
) -> SDRAgentConfig:
    return SDRAgentConfig(
        lead_id=lead_id,
        calendar_id=calendar_id,
        sales_rep_name=sales_rep_name,
        initial_message_text=initial_message_text,
        # RespondAgent subclasses must use respond(), not generate_response().
        generate_responses=False,
        allow_agent_to_be_cut_off=allow_agent_to_be_cut_off,
        interrupt_sensitivity=interrupt_sensitivity,
        end_conversation_on_goodbye=True,
    )
