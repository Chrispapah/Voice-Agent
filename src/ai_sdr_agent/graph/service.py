from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

from ai_sdr_agent.graph.graph import build_sdr_graph
from ai_sdr_agent.graph.state import ConversationState
from ai_sdr_agent.models import CallLogRecord
from ai_sdr_agent.services.brain import ConversationBrain
from ai_sdr_agent.services.persistence import CallLogRepository, SessionStore
from ai_sdr_agent.services.pre_call_loader import PreCallLoader
from ai_sdr_agent.tools import CRMGateway, CalendarGateway, EmailGateway


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
        return updated_state
