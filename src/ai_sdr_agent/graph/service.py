from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from ai_sdr_agent.graph.graph import build_sdr_graph
from ai_sdr_agent.graph.state import ConversationState
from ai_sdr_agent.models import CallLogRecord
from ai_sdr_agent.services.brain import ConversationBrain
from ai_sdr_agent.services.persistence import CallLogRepository, SessionStore
from ai_sdr_agent.services.pre_call_loader import PreCallLoader
from ai_sdr_agent.tools import CRMGateway, CalendarGateway, EmailGateway

_EXIT_PATTERNS = re.compile(
    r"\b("
    r"goodbye|good bye|bye bye|hang up|end the call|end call|stop calling"
    r"|leave me alone|go away|get lost|piss off|fuck off|fuck you"
    r"|screw you|shut up|stop it|i('?m| am) done|let me go"
    r"|not interested|remove me|do not call|don'?t call"
    r")\b",
    re.IGNORECASE,
)

MAX_CALL_TURNS = 12


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


class SDRConversationService:
    def __init__(self, dependencies: SDRRuntimeDependencies):
        self.dependencies = dependencies
        self.graph = build_sdr_graph(
            brain=dependencies.brain,
            calendar_gateway=dependencies.calendar_gateway,
            email_gateway=dependencies.email_gateway,
            crm_gateway=dependencies.crm_gateway,
            email_template_path=dependencies.email_template_path,
            sales_rep_name=dependencies.sales_rep_name,
            from_name=dependencies.from_name,
        )

    async def start_session(self, lead_id: str, *, conversation_id: str | None = None) -> str:
        session_id = conversation_id or f"conv-{uuid.uuid4().hex[:12]}"
        state = await self.dependencies.pre_call_loader.build_initial_state(lead_id)
        await self.dependencies.session_store.save(session_id, state)
        await self.dependencies.call_log_repository.save_call_log(
            CallLogRecord(
                conversation_id=session_id,
                lead_id=lead_id,
                transcript=[],
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

        force_exit = False
        if human_input and _EXIT_PATTERNS.search(human_input):
            force_exit = True
            logger.info("Exit signal detected conversation_id={} text={!r}", conversation_id, human_input)
        if state["turn_count"] >= MAX_CALL_TURNS:
            force_exit = True
            logger.info("Max turns reached conversation_id={} turn_count={}", conversation_id, state["turn_count"])
        if force_exit and state["next_node"] != "complete":
            state["next_node"] = "wrap_up"
            state["call_outcome"] = "not_interested"

        updated_state = await self.graph.ainvoke(state)
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
        logger.info(
            "Completed SDR turn conversation_id={} route_decision={} response={!r}",
            conversation_id,
            updated_state["route_decision"],
            updated_state["last_agent_response"],
        )
        return updated_state
