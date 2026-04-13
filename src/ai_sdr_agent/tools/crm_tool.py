from __future__ import annotations

from typing import Protocol

from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from ai_sdr_agent.models import LeadRecord


class CRMUpdateRequest(BaseModel):
    lead_id: str
    call_outcome: str
    notes: str


class CRMGateway(Protocol):
    async def get_lead_context(self, lead_id: str) -> LeadRecord:
        ...

    async def update_call_outcome(
        self,
        *,
        lead_id: str,
        call_outcome: str,
        notes: str,
    ) -> dict[str, str]:
        ...


class StubCRMGateway:
    def __init__(self, seed_leads: list[LeadRecord]):
        self._leads = {lead.lead_id: lead for lead in seed_leads}
        self.updates: list[dict[str, str]] = []

    async def get_lead_context(self, lead_id: str) -> LeadRecord:
        lead = self._leads.get(lead_id)
        if lead is None:
            raise KeyError(f"Unknown lead_id: {lead_id}")
        return lead

    async def update_call_outcome(
        self,
        *,
        lead_id: str,
        call_outcome: str,
        notes: str,
    ) -> dict[str, str]:
        payload = {
            "lead_id": lead_id,
            "call_outcome": call_outcome,
            "notes": notes,
        }
        self.updates.append(payload)
        return payload


def build_crm_tools(crm_gateway: CRMGateway) -> list[StructuredTool]:
    async def update_crm(lead_id: str, call_outcome: str, notes: str) -> dict[str, str]:
        return await crm_gateway.update_call_outcome(
            lead_id=lead_id,
            call_outcome=call_outcome,
            notes=notes,
        )

    return [
        StructuredTool.from_function(
            coroutine=update_crm,
            name="update_crm",
            description="Update the CRM with the SDR call outcome and notes.",
            args_schema=CRMUpdateRequest,
        )
    ]
