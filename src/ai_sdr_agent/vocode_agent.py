from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field

from loguru import logger
from vocode.streaming.agent.base_agent import AgentResponseMessage, RespondAgent
from vocode.streaming.models.actions import EndOfTurn
from vocode.streaming.models.agent import AgentConfig
from vocode.streaming.models.message import BaseMessage

from ai_sdr_agent.graph.service import SDRConversationService
from ai_sdr_agent.services.latency_analytics import (
    clear_phone_turn_on_error,
    mark_phone_turn_graph_done,
    mark_phone_turn_respond_enter,
)


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
class _QueuedTurnResult:
    response: str | None
    should_stop: bool
    skip_reason: str | None = None


@dataclass
class _QueuedTurn:
    text: str
    normalized: str
    is_interrupt: bool
    sequence: int | None
    received_at: float
    result_future: asyncio.Future[_QueuedTurnResult]


@dataclass
class _ConversationRuntimeState:
    state_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    work_ready: asyncio.Event = field(default_factory=asyncio.Event)
    queued_turns: deque[_QueuedTurn] = field(default_factory=deque)
    active_turn: _QueuedTurn | None = None
    latest_interrupt_sequence: int = 0
    worker_task: asyncio.Task[None] | None = None


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
    def _resolve_turn_result(
        turn: _QueuedTurn,
        result: _QueuedTurnResult,
    ) -> None:
        if not turn.result_future.done():
            turn.result_future.set_result(result)

    def _ensure_worker(
        self,
        conversation_id: str,
        runtime_state: _ConversationRuntimeState,
    ) -> None:
        worker_task = runtime_state.worker_task
        if worker_task is not None and not worker_task.done():
            return
        runtime_state.worker_task = asyncio.create_task(
            self._run_conversation_worker(conversation_id, runtime_state)
        )
        logger.info("Started conversation turn worker conversation_id={}", conversation_id)

    async def _enqueue_turn(
        self,
        conversation_id: str,
        human_input: str,
        *,
        is_interrupt: bool,
        runtime_state: _ConversationRuntimeState,
    ) -> tuple[_QueuedTurn | None, str | None]:
        loop = asyncio.get_running_loop()
        normalized_input = _normalize_interrupt_text(human_input)
        async with runtime_state.state_lock:
            if is_interrupt:
                if not normalized_input:
                    _log_interrupt_buffer_event(
                        conversation_id,
                        "drop_empty_interrupt",
                    )
                    return None, "empty_interrupt"
                active_turn = runtime_state.active_turn
                if active_turn is not None and active_turn.normalized == normalized_input:
                    _log_interrupt_buffer_event(
                        conversation_id,
                        "drop_active_repeat_interrupt",
                        active_text=active_turn.text,
                        text=human_input,
                    )
                    return None, "duplicate_active_input"
                last_queued_turn = (
                    runtime_state.queued_turns[-1] if runtime_state.queued_turns else None
                )
                if last_queued_turn is not None and last_queued_turn.is_interrupt:
                    if last_queued_turn.normalized == normalized_input:
                        _log_interrupt_buffer_event(
                            conversation_id,
                            "drop_repeated_interrupt",
                            text=human_input,
                        )
                        return None, "duplicate_interrupt"
                    runtime_state.queued_turns.pop()
                    self._resolve_turn_result(
                        last_queued_turn,
                        _QueuedTurnResult(
                            response=None,
                            should_stop=False,
                            skip_reason="stale_interrupt",
                        ),
                    )
                    runtime_state.latest_interrupt_sequence += 1
                    turn = _QueuedTurn(
                        text=human_input,
                        normalized=normalized_input,
                        is_interrupt=True,
                        sequence=runtime_state.latest_interrupt_sequence,
                        received_at=time.perf_counter(),
                        result_future=loop.create_future(),
                    )
                    runtime_state.queued_turns.append(turn)
                    _log_interrupt_buffer_event(
                        conversation_id,
                        "replace_pending_interrupt",
                        sequence=turn.sequence,
                        previous_text=last_queued_turn.text,
                        text=human_input,
                    )
                else:
                    runtime_state.latest_interrupt_sequence += 1
                    turn = _QueuedTurn(
                        text=human_input,
                        normalized=normalized_input,
                        is_interrupt=True,
                        sequence=runtime_state.latest_interrupt_sequence,
                        received_at=time.perf_counter(),
                        result_future=loop.create_future(),
                    )
                    runtime_state.queued_turns.append(turn)
                    _log_interrupt_buffer_event(
                        conversation_id,
                        "buffer_interrupt",
                        sequence=turn.sequence,
                        text=human_input,
                    )
            else:
                turn = _QueuedTurn(
                    text=human_input,
                    normalized=normalized_input,
                    is_interrupt=False,
                    sequence=None,
                    received_at=time.perf_counter(),
                    result_future=loop.create_future(),
                )
                runtime_state.queued_turns.append(turn)
                logger.info(
                    "Queued turn conversation_id={} is_interrupt={} queue_depth={} text={!r}",
                    conversation_id,
                    is_interrupt,
                    len(runtime_state.queued_turns),
                    human_input,
                )
            runtime_state.work_ready.set()
            self._ensure_worker(conversation_id, runtime_state)
            return turn, None

    async def _process_turn(
        self,
        conversation_id: str,
        turn: _QueuedTurn,
        *,
        remaining_turns: int,
    ) -> _QueuedTurnResult:
        queue_wait_ms = (time.perf_counter() - turn.received_at) * 1000
        _log_agent_step_latency(
            conversation_id,
            "respond_queue_wait",
            queue_wait_ms,
            is_interrupt=turn.is_interrupt,
            sequence=turn.sequence,
            remaining_turns=remaining_turns,
        )
        if turn.is_interrupt:
            _log_interrupt_buffer_event(
                conversation_id,
                "flush_interrupt",
                sequence=turn.sequence,
                text=turn.text,
            )
        mark_phone_turn_respond_enter(conversation_id)
        try:
            if conversation_id not in self._initialized_conversations:
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
            handle_turn_t0 = time.perf_counter()
            state = await self.conversation_service.handle_turn(conversation_id, turn.text)
            handle_turn_ms = (time.perf_counter() - handle_turn_t0) * 1000
            _log_agent_step_latency(
                conversation_id,
                "handle_turn",
                handle_turn_ms,
                turn_count=state["turn_count"],
                route_decision=state["route_decision"],
            )
            response = state["last_agent_response"]
            if not response:
                total_ms = (time.perf_counter() - turn.received_at) * 1000
                _log_agent_step_latency(conversation_id, "respond_total", total_ms)
                logger.info("Conversation complete, no reply conversation_id={}", conversation_id)
                clear_phone_turn_on_error(conversation_id)
                return _QueuedTurnResult(response=None, should_stop=True)
            mark_phone_turn_graph_done(conversation_id)
            total_ms = (time.perf_counter() - turn.received_at) * 1000
            _log_agent_step_latency(
                conversation_id,
                "respond_total",
                total_ms,
                turn_count=state["turn_count"],
            )
            logger.info(
                "SDR agent generated reply conversation_id={} text={!r}",
                conversation_id,
                response,
            )
            return _QueuedTurnResult(response=response, should_stop=False)
        except Exception:
            total_ms = (time.perf_counter() - turn.received_at) * 1000
            _log_agent_step_latency(
                conversation_id,
                "respond_failed",
                total_ms,
                is_interrupt=turn.is_interrupt,
                sequence=turn.sequence,
            )
            logger.exception(
                "SDR agent respond() failed conversation_id={} text={!r}",
                conversation_id,
                turn.text,
            )
            mark_phone_turn_graph_done(conversation_id)
            return _QueuedTurnResult(
                response="I'm sorry, I'm having a technical issue. Could you hold on a moment?",
                should_stop=False,
            )

    async def _run_conversation_worker(
        self,
        conversation_id: str,
        runtime_state: _ConversationRuntimeState,
    ) -> None:
        current_task = asyncio.current_task()
        try:
            while True:
                await runtime_state.work_ready.wait()
                while True:
                    async with runtime_state.state_lock:
                        if not runtime_state.queued_turns:
                            runtime_state.work_ready.clear()
                            break
                        turn = runtime_state.queued_turns.popleft()
                        runtime_state.active_turn = turn
                        remaining_turns = len(runtime_state.queued_turns)
                    try:
                        result = await self._process_turn(
                            conversation_id,
                            turn,
                            remaining_turns=remaining_turns,
                        )
                    finally:
                        async with runtime_state.state_lock:
                            if runtime_state.active_turn is turn:
                                runtime_state.active_turn = None
                    self._resolve_turn_result(turn, result)
        except asyncio.CancelledError:
            logger.warning("Conversation turn worker cancelled conversation_id={}", conversation_id)
            async with runtime_state.state_lock:
                active_turn = runtime_state.active_turn
                queued_turns = list(runtime_state.queued_turns)
                runtime_state.active_turn = None
                runtime_state.queued_turns.clear()
                runtime_state.work_ready.clear()
            turns_to_cancel = queued_turns
            if active_turn is not None:
                turns_to_cancel.insert(0, active_turn)
            for pending_turn in turns_to_cancel:
                if not pending_turn.result_future.done():
                    pending_turn.result_future.cancel()
            raise
        finally:
            async with runtime_state.state_lock:
                if runtime_state.worker_task is current_task:
                    runtime_state.worker_task = None

    def start(self) -> None:
        super().start()
        logger.info(
            "SDRVocodeAgent started lead_id={} generate_responses={}",
            self.agent_config.lead_id,
            self.agent_config.generate_responses,
        )
        self.produce_interruptible_agent_response_event_nonblocking(
            AgentResponseMessage(
                message=BaseMessage(text=self.agent_config.initial_message_text),
                is_first=True,
            ),
            is_interruptible=self.agent_config.allow_agent_to_be_cut_off,
        )
        self.produce_interruptible_agent_response_event_nonblocking(
            AgentResponseMessage(message=EndOfTurn()),
            is_interruptible=self.agent_config.allow_agent_to_be_cut_off,
        )

    async def respond(
        self,
        human_input: str,
        conversation_id: str,
        is_interrupt: bool = False,
    ) -> tuple[str | None, bool]:
        respond_start = time.perf_counter()
        runtime_state = self._get_runtime_state(conversation_id)
        logger.info(
            "SDR agent received speech conversation_id={} is_interrupt={} text={!r}",
            conversation_id,
            is_interrupt,
            human_input,
        )
        turn, skip_reason = await self._enqueue_turn(
            conversation_id,
            human_input,
            is_interrupt=is_interrupt,
            runtime_state=runtime_state,
        )
        if turn is None:
            total_ms = (time.perf_counter() - respond_start) * 1000
            _log_agent_step_latency(
                conversation_id,
                "respond_coalesced_skip",
                total_ms,
                reason=skip_reason,
            )
            return None, False
        try:
            result = await asyncio.shield(turn.result_future)
        except asyncio.CancelledError:
            total_ms = (time.perf_counter() - respond_start) * 1000
            _log_agent_step_latency(
                conversation_id,
                "respond_wait_cancelled",
                total_ms,
                is_interrupt=is_interrupt,
                sequence=turn.sequence,
            )
            raise
        if result.skip_reason is not None:
            total_ms = (time.perf_counter() - respond_start) * 1000
            _log_agent_step_latency(
                conversation_id,
                "respond_coalesced_skip",
                total_ms,
                reason=result.skip_reason,
                sequence=turn.sequence,
            )
        return result.response, result.should_stop


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
