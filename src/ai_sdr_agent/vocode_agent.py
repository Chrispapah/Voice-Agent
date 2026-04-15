from __future__ import annotations

import time

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
        logger.info(
            "SDR agent received speech conversation_id={} is_interrupt={} text={!r}",
            conversation_id,
            is_interrupt,
            human_input,
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
            state = await self.conversation_service.handle_turn(conversation_id, human_input)
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
                total_ms = (time.perf_counter() - respond_start) * 1000
                _log_agent_step_latency(conversation_id, "respond_total", total_ms)
                logger.info("Conversation complete, no reply conversation_id={}", conversation_id)
                clear_phone_turn_on_error(conversation_id)
                return None, True
            mark_phone_turn_graph_done(conversation_id)
            total_ms = (time.perf_counter() - respond_start) * 1000
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
            return response, False
        except Exception:
            total_ms = (time.perf_counter() - respond_start) * 1000
            _log_agent_step_latency(conversation_id, "respond_failed", total_ms)
            logger.exception(
                "SDR agent respond() failed conversation_id={} text={!r}",
                conversation_id,
                human_input,
            )
            mark_phone_turn_graph_done(conversation_id)
            return "I'm sorry, I'm having a technical issue. Could you hold on a moment?", False


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
