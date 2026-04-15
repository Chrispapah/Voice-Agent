from __future__ import annotations

import asyncio
import time
import re

from loguru import logger
from vocode.streaming.agent.base_agent import AgentResponseMessage, RespondAgent
from vocode.streaming.models.actions import EndOfTurn
from vocode.streaming.models.agent import AgentConfig
from vocode.streaming.models.message import BaseMessage, BotBackchannel
from vocode.streaming.utils import unrepeating_randomizer

from ai_sdr_agent.config import DEFAULT_AGENT_PREFILL_ACK_PHRASES
from ai_sdr_agent.graph.service import SDRConversationService
from ai_sdr_agent.services.latency_analytics import (
    clear_phone_turn_on_error,
    mark_phone_turn_graph_done,
    mark_phone_turn_respond_enter,
)


class SDRAgentConfig(AgentConfig, type="agent_sdr"):  # type: ignore[misc]
    lead_id: str
    calendar_id: str
    sales_rep_name: str
    initial_message_text: str
    prefill_ack_enabled: bool = False
    prefill_ack_phrases: tuple[str, ...] = DEFAULT_AGENT_PREFILL_ACK_PHRASES
    prefill_ack_on_safe_interrupts: bool = True


_DUPLICATE_SHORT_INTERRUPT_WINDOW_S = 1.5
_DUPLICATE_SHORT_INTERRUPT_MAX_WORDS = 2
_DUPLICATE_SHORT_INTERRUPT_MAX_CHARS = 24
_SAFE_INTERRUPT_ACK_PATTERN = re.compile(
    r"^(yes|yeah|yep|ok|okay|sure|go ahead|please do|sounds good|mm hmm|mhm|uh huh)$",
    re.IGNORECASE,
)


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
        self._inflight_turns: dict[str, asyncio.Task[tuple[str | None, bool]]] = {}
        self._recent_short_interrupts: dict[str, tuple[str, float]] = {}
        self._respond_locks: dict[str, asyncio.Lock] = {}
        phrases = list(agent_config.prefill_ack_phrases) or list(DEFAULT_AGENT_PREFILL_ACK_PHRASES)
        if len(phrases) >= 2:
            self._prefill_ack_pick = unrepeating_randomizer(phrases)
        else:
            only = phrases[0]
            self._prefill_ack_pick = lambda: only

    def _get_respond_lock(self, conversation_id: str) -> asyncio.Lock:
        lock = self._respond_locks.get(conversation_id)
        if lock is None:
            lock = asyncio.Lock()
            self._respond_locks[conversation_id] = lock
        return lock

    @staticmethod
    def _normalize_interrupt_text(text: str) -> str:
        return " ".join(text.lower().split())

    @staticmethod
    def _is_short_interrupt_text(text: str) -> bool:
        words = text.split()
        return 0 < len(words) <= _DUPLICATE_SHORT_INTERRUPT_MAX_WORDS and len(text) <= _DUPLICATE_SHORT_INTERRUPT_MAX_CHARS

    def _should_skip_duplicate_short_interrupt(
        self,
        *,
        conversation_id: str,
        human_input: str,
        is_interrupt: bool,
    ) -> bool:
        normalized = self._normalize_interrupt_text(human_input)
        if not normalized or not self._is_short_interrupt_text(normalized):
            return False
        now = time.monotonic()
        previous = self._recent_short_interrupts.get(conversation_id)
        # Remember short utterances even when the first transcript was not marked as
        # an interrupt so a duplicated replay can still be suppressed if the repeat
        # arrives moments later as an interrupt.
        self._recent_short_interrupts[conversation_id] = (normalized, now)
        if not is_interrupt or previous is None:
            return False
        last_text, last_ts = previous
        return last_text == normalized and (now - last_ts) <= _DUPLICATE_SHORT_INTERRUPT_WINDOW_S

    def _should_allow_prefill_for_interrupt(self, human_input: str, is_interrupt: bool) -> bool:
        if not is_interrupt:
            return True
        if not self.agent_config.prefill_ack_on_safe_interrupts:
            return False
        normalized = self._normalize_interrupt_text(human_input)
        if not normalized or not self._is_short_interrupt_text(normalized):
            return False
        return _SAFE_INTERRUPT_ACK_PATTERN.fullmatch(normalized) is not None

    @staticmethod
    def _clear_current_task_cancellation_state() -> int:
        current_task = asyncio.current_task()
        if current_task is None:
            return 0
        cleared = 0
        while current_task.cancelling():
            current_task.uncancel()
            cleared += 1
        return cleared

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
        if self._should_skip_duplicate_short_interrupt(
            conversation_id=conversation_id,
            human_input=human_input,
            is_interrupt=is_interrupt,
        ):
            logger.info(
                "Skipping duplicate short interrupt transcript conversation_id={} text={!r}",
                conversation_id,
                human_input,
            )
            return None, False
        inflight_turn = self._inflight_turns.get(conversation_id)
        if is_interrupt and inflight_turn is not None and not inflight_turn.done():
            logger.info(
                "Skipping interrupt transcript while SDR turn is already running "
                "conversation_id={} text={!r}",
                conversation_id,
                human_input,
            )
            return None, False
        async with self._get_respond_lock(conversation_id):
            if is_interrupt:
                inflight_turn = self._inflight_turns.get(conversation_id)
                if inflight_turn is not None and not inflight_turn.done():
                    logger.info(
                        "Skipping interrupt transcript after acquiring respond lock "
                        "conversation_id={} text={!r}",
                        conversation_id,
                        human_input,
                    )
                    return None, False

            turn_task = asyncio.create_task(
                self._run_turn(
                    human_input=human_input,
                    conversation_id=conversation_id,
                    is_interrupt=is_interrupt,
                )
            )
            self._inflight_turns[conversation_id] = turn_task
            try:
                while True:
                    try:
                        return await asyncio.shield(turn_task)
                    except asyncio.CancelledError:
                        if turn_task.done():
                            cleared = self._clear_current_task_cancellation_state()
                            logger.warning(
                                "SDR agent respond() caught cancellation after turn completed "
                                "conversation_id={} cleared_cancellations={}",
                                conversation_id,
                                cleared,
                            )
                            return await turn_task
                        cleared = self._clear_current_task_cancellation_state()
                        logger.warning(
                            "SDR agent respond() cancellation ignored while preserving in-flight turn "
                            "conversation_id={} text={!r} cleared_cancellations={}",
                            conversation_id,
                            human_input,
                            cleared,
                        )
            finally:
                if self._inflight_turns.get(conversation_id) is turn_task and turn_task.done():
                    self._inflight_turns.pop(conversation_id, None)

    async def _run_turn(
        self,
        *,
        human_input: str,
        conversation_id: str,
        is_interrupt: bool,
    ) -> tuple[str | None, bool]:
        logger.info(
            "SDR agent received speech conversation_id={} is_interrupt={} text={!r}",
            conversation_id,
            is_interrupt,
            human_input,
        )
        mark_phone_turn_respond_enter(conversation_id)
        try:
            if conversation_id not in self._initialized_conversations:
                await self.conversation_service.start_session(
                    self.agent_config.lead_id,
                    conversation_id=conversation_id,
                    initial_agent_message=self.agent_config.initial_message_text,
                    initial_current_node="greeting",
                    initial_next_node="qualify_lead",
                )
                self._initialized_conversations.add(conversation_id)
                logger.info(
                    "Initialized SDR session from phone call conversation_id={} lead_id={}",
                    conversation_id,
                    self.agent_config.lead_id,
                )
            prefill_emitted = await self._maybe_emit_prefill_ack(
                conversation_id=conversation_id,
                human_input=human_input,
                is_interrupt=is_interrupt,
            )
            state = await self.conversation_service.handle_turn(conversation_id, human_input)
            response = state["last_agent_response"]
            if not response:
                # Prefill should not leave users in dead air when a turn resolves with no main
                # text response (for example, conversation transitioning to complete).
                if prefill_emitted:
                    fallback_text = "Thanks for your time. Goodbye."
                    logger.warning(
                        "Prefill emitted without main reply conversation_id={} next_node={} - using fallback",
                        conversation_id,
                        state.get("next_node"),
                    )
                    clear_phone_turn_on_error(conversation_id)
                    return fallback_text, state.get("next_node") == "complete"
                logger.info("Conversation complete, no reply conversation_id={}", conversation_id)
                clear_phone_turn_on_error(conversation_id)
                return None, True
            mark_phone_turn_graph_done(conversation_id)
            logger.info(
                "SDR agent generated reply conversation_id={} text={!r}",
                conversation_id,
                response,
            )
            return response, False
        except asyncio.CancelledError:
            logger.warning(
                "SDR agent _run_turn() cancelled conversation_id={} text={!r}",
                conversation_id,
                human_input,
            )
            clear_phone_turn_on_error(conversation_id)
            raise
        except Exception:
            logger.exception(
                "SDR agent respond() failed conversation_id={} text={!r}",
                conversation_id,
                human_input,
            )
            mark_phone_turn_graph_done(conversation_id)
            return "I'm sorry, I'm having a technical issue. Could you hold on a moment?", False

    async def _maybe_emit_prefill_ack(
        self,
        *,
        conversation_id: str,
        human_input: str,
        is_interrupt: bool,
    ) -> bool:
        if not self.agent_config.prefill_ack_enabled:
            logger.debug("Prefill ack skipped (disabled) conversation_id={}", conversation_id)
            return False
        if not human_input.strip():
            logger.debug("Prefill ack skipped (empty input) conversation_id={}", conversation_id)
            return False
        if not self._should_allow_prefill_for_interrupt(human_input, is_interrupt):
            logger.debug(
                "Prefill ack skipped (interrupt not safe for ack) conversation_id={} text={!r}",
                conversation_id,
                human_input,
            )
            return False
        pre_state = await self.conversation_service.get_state(conversation_id)
        if pre_state["next_node"] == "complete":
            logger.debug("Prefill ack skipped (already complete) conversation_id={}", conversation_id)
            return False
        phrase = self._prefill_ack_pick()
        logger.debug(
            "Prefill ack emitting conversation_id={} phrase={!r}",
            conversation_id,
            phrase,
        )
        # Always keep the latency-hiding backchannel easy to barge into so it never
        # blocks the caller or delays the main graph reply.
        self.produce_interruptible_agent_response_event_nonblocking(
            AgentResponseMessage(message=BotBackchannel(text=phrase)),
            is_interruptible=True,
        )
        return True


def build_agent_config(
    *,
    lead_id: str,
    calendar_id: str,
    sales_rep_name: str,
    initial_message_text: str,
    allow_agent_to_be_cut_off: bool = True,
    interrupt_sensitivity: str = "low",
    prefill_ack_enabled: bool = False,
    prefill_ack_phrases: tuple[str, ...] = DEFAULT_AGENT_PREFILL_ACK_PHRASES,
    prefill_ack_on_safe_interrupts: bool = True,
) -> SDRAgentConfig:
    return SDRAgentConfig(
        lead_id=lead_id,
        calendar_id=calendar_id,
        sales_rep_name=sales_rep_name,
        initial_message_text=initial_message_text,
        prefill_ack_enabled=prefill_ack_enabled,
        prefill_ack_phrases=prefill_ack_phrases,
        prefill_ack_on_safe_interrupts=prefill_ack_on_safe_interrupts,
        # RespondAgent subclasses must use respond(), not generate_response().
        generate_responses=False,
        allow_agent_to_be_cut_off=allow_agent_to_be_cut_off,
        interrupt_sensitivity=interrupt_sensitivity,
        end_conversation_on_goodbye=True,
    )
