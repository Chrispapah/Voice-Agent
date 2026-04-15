from __future__ import annotations

import asyncio
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from ai_sdr_agent.graph.graph import build_sdr_graph
from ai_sdr_agent.graph.state import ConversationState
from ai_sdr_agent.models import CallLogRecord
from ai_sdr_agent.services.brain import ConversationBrain
from ai_sdr_agent.services.latency_analytics import LatencyAnalyticsBuffer
from ai_sdr_agent.services.persistence import CallLogRepository, SessionStore
from ai_sdr_agent.services.pre_call_loader import PreCallLoader
from ai_sdr_agent.tools import CRMGateway, CalendarGateway, EmailGateway

_EXIT_PATTERNS = re.compile(
    r"\b("
    r"goodbye|good bye|bye bye|hang up|end the call|end call|stop calling"
    r"|leave me alone|go away|get lost(?!\s+in\b)|piss off|fuck off|fuck you"
    r"|screw you|shut up|stop it|i('?m| am) done|let me go"
    r"|not interested|remove me|do not call|don'?t call"
    r"|no thank you|no thanks|no thankyou"
    r"|nah|nope"
    r")\b",
    re.IGNORECASE,
)

_DEFAULT_MAX_CALL_TURNS = 12


@dataclass
class SDRRuntimeDependencies:
    brain: ConversationBrain
    calendar_gateway: CalendarGateway
    email_gateway: EmailGateway
    crm_gateway: CRMGateway
    pre_call_loader: PreCallLoader
    session_store: SessionStore
    call_log_repository: CallLogRepository
    email_template_path: Path
    sales_rep_name: str
    from_name: str
    latency_analytics: LatencyAnalyticsBuffer | None = None


class SDRConversationService:
    def __init__(self, dependencies: SDRRuntimeDependencies, bot_config: dict | None = None):
        self.dependencies = dependencies
        self._bot_config = bot_config
        self._turn_locks: dict[str, asyncio.Lock] = {}
        self.graph = build_sdr_graph(
            brain=dependencies.brain,
            calendar_gateway=dependencies.calendar_gateway,
            email_gateway=dependencies.email_gateway,
            crm_gateway=dependencies.crm_gateway,
            email_template_path=dependencies.email_template_path,
            sales_rep_name=dependencies.sales_rep_name,
            from_name=dependencies.from_name,
        )

    def _get_turn_lock(self, conversation_id: str) -> asyncio.Lock:
        lock = self._turn_locks.get(conversation_id)
        if lock is None:
            lock = asyncio.Lock()
            self._turn_locks[conversation_id] = lock
        return lock

    async def start_session(
        self,
        lead_id: str,
        *,
        conversation_id: str | None = None,
        bot_config: dict | None = None,
        initial_agent_message: str | None = None,
        initial_current_node: str | None = None,
        initial_next_node: str | None = None,
    ) -> str:
        session_id = conversation_id or f"conv-{uuid.uuid4().hex[:12]}"
        cfg = bot_config or self._bot_config
        state = await self.dependencies.pre_call_loader.build_initial_state(lead_id, bot_config=cfg)
        if initial_agent_message:
            seeded_next_node = initial_next_node or state["next_node"]
            state = {
                **state,
                "transcript": state["transcript"] + [{"role": "agent", "content": initial_agent_message}],
                "current_node": initial_current_node or state["current_node"],
                "last_agent_response": initial_agent_message,
                "next_node": seeded_next_node,
                "route_decision": seeded_next_node,
            }
        await self.dependencies.session_store.save(session_id, state)
        await self.dependencies.call_log_repository.save_call_log(
            CallLogRecord(
                conversation_id=session_id,
                lead_id=lead_id,
                transcript=state["transcript"],
            )
        )
        logger.info("Started SDR session conversation_id={} lead_id={}", session_id, lead_id)
        return session_id

    async def get_state(self, conversation_id: str) -> ConversationState:
        state = await self.dependencies.session_store.get(conversation_id)
        if state is None:
            raise KeyError(f"Unknown conversation_id: {conversation_id}")
        return state

    async def handle_turn(self, conversation_id: str, human_input: str) -> ConversationState:
        async with self._get_turn_lock(conversation_id):
            state = await self.get_state(conversation_id)
            if human_input:
                state["transcript"].append({"role": "human", "content": human_input})
                state["last_human_message"] = human_input
                state["turn_count"] += 1
                logger.info(
                    "Processing human turn conversation_id={} turn_count={} text={!r}",
                    conversation_id,
                    state["turn_count"],
                    human_input,
                )
            else:
                logger.info("Processing empty turn conversation_id={}", conversation_id)

            if state["next_node"] == "complete":
                logger.info("Conversation already complete conversation_id={}", conversation_id)
                state["last_agent_response"] = ""
                await self.dependencies.session_store.save(conversation_id, state)
                return state

            force_exit = False
            if human_input and _EXIT_PATTERNS.search(human_input):
                force_exit = True
                logger.info("Exit signal detected conversation_id={} text={!r}", conversation_id, human_input)
            max_turns = state.get("bot_config", {}).get("max_call_turns", _DEFAULT_MAX_CALL_TURNS)
            if state["turn_count"] >= max_turns:
                force_exit = True
                logger.info(
                    "Max turns reached conversation_id={} turn_count={}",
                    conversation_id,
                    state["turn_count"],
                )
            if force_exit and state["next_node"] != "complete":
                state["next_node"] = "wrap_up"
                state["call_outcome"] = "not_interested"

            turn_start = time.perf_counter()
            updated_state = await self.graph.ainvoke(state)
            graph_ms = (time.perf_counter() - turn_start) * 1000

            persist_start = time.perf_counter()
            await self.dependencies.session_store.save(conversation_id, updated_state)
            await self.dependencies.call_log_repository.save_call_log(
                CallLogRecord(
                    conversation_id=conversation_id,
                    lead_id=updated_state["lead_id"],
                    call_outcome=updated_state["call_outcome"],
                    transcript=updated_state["transcript"],
                    qualification_notes=updated_state["qualification_notes"],
                    meeting_booked=updated_state["meeting_booked"],
                    proposed_slot=updated_state["proposed_slot"],
                    follow_up_action=updated_state["follow_up_action"],
                )
            )
            persist_ms = (time.perf_counter() - persist_start) * 1000
            total_ms = (time.perf_counter() - turn_start) * 1000

            logger.info(
                "Completed SDR turn conversation_id={} route_decision={} "
                "latency_total_ms={:.0f} latency_graph_ms={:.0f} latency_persist_ms={:.0f} "
                "response={!r}",
                conversation_id,
                updated_state["route_decision"],
                total_ms,
                graph_ms,
                persist_ms,
                updated_state["last_agent_response"],
            )
            la = self.dependencies.latency_analytics
            if la is not None:
                await la.record_turn(
                    conversation_id=conversation_id,
                    turn_count=int(updated_state["turn_count"]),
                    route_decision=str(updated_state["route_decision"]),
                    latency_total_ms=total_ms,
                    latency_graph_ms=graph_ms,
                    latency_persist_ms=persist_ms,
                )
            return updated_state
