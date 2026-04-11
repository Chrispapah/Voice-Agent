from __future__ import annotations

from ai_sdr_agent.graph.state import ConversationState, build_initial_state
from ai_sdr_agent.models import LeadRecord
from ai_sdr_agent.services.persistence import LeadRepository
from ai_sdr_agent.tools import CalendarGateway


class PreCallLoader:
    def __init__(
        self,
        *,
        lead_repository: LeadRepository,
        calendar_gateway: CalendarGateway,
    ):
        self.lead_repository = lead_repository
        self.calendar_gateway = calendar_gateway

    async def load_lead(self, lead_id: str) -> LeadRecord:
        return await self.lead_repository.get_lead(lead_id)

    async def build_initial_state(self, lead_id: str) -> ConversationState:
        lead = await self.load_lead(lead_id)
        slots = await self.calendar_gateway.list_available_slots(
            calendar_id=lead.calendar_id,
            date_range="next_5_business_days",
        )
        return build_initial_state(
            lead_id=lead.lead_id,
            lead_name=lead.lead_name,
            lead_email=str(lead.lead_email),
            phone_number=lead.phone_number,
            company=lead.company,
            calendar_id=lead.calendar_id,
            lead_context=lead.lead_context,
            available_slots=[
                {
                    "slot_id": slot.slot_id,
                    "start_time": slot.start_time.isoformat(),
                    "end_time": slot.end_time.isoformat(),
                    "label": slot.label,
                }
                for slot in slots
            ],
        )
