from __future__ import annotations

from ai_sdr_agent.graph.state import BotConfigDict, ConversationState, build_initial_state
from ai_sdr_agent.models import LeadRecord
from ai_sdr_agent.services.persistence import LeadRepository


class PreCallLoader:
    def __init__(
        self,
        *,
        lead_repository: LeadRepository,
    ):
        self.lead_repository = lead_repository

    async def load_lead(self, lead_id: str) -> LeadRecord:
        return await self.lead_repository.get_lead(lead_id)

    async def build_initial_state(
        self,
        lead_id: str,
        bot_config: BotConfigDict | None = None,
    ) -> ConversationState:
        lead = await self.load_lead(lead_id)
        return build_initial_state(
            lead_id=lead.lead_id,
            lead_name=lead.lead_name,
            lead_email=str(lead.lead_email),
            phone_number=lead.phone_number,
            company=lead.company,
            calendar_id=lead.calendar_id,
            lead_context=lead.lead_context,
            bot_config=bot_config,
        )
