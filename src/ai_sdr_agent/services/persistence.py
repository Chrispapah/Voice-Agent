from __future__ import annotations

import copy
from typing import Protocol

from ai_sdr_agent.models import CallLogRecord, LeadRecord


class LeadRepository(Protocol):
    async def get_lead(self, lead_id: str) -> LeadRecord:
        ...

    async def update_lead_status(self, lead_id: str, *, lifecycle_stage: str) -> LeadRecord:
        ...


class CallLogRepository(Protocol):
    async def save_call_log(self, call_log: CallLogRecord) -> CallLogRecord:
        ...

    async def get_call_log(self, conversation_id: str) -> CallLogRecord | None:
        ...


class SessionStore(Protocol):
    async def get(self, conversation_id: str) -> dict | None:
        ...

    async def save(self, conversation_id: str, state: dict) -> None:
        ...

    async def delete(self, conversation_id: str) -> None:
        ...


class InMemoryLeadRepository:
    def __init__(self, leads: list[LeadRecord] | None = None):
        seed = leads or [
            LeadRecord(
                lead_id="lead-001",
                lead_name="Michael Manis",
                company="HarborLab",
                phone_number="+15551234567",
                lead_email="michael.manis@harborlab.com",
                lead_context=(
                    "Downloaded the outbound sales playbook last week and requested "
                    "a follow-up about routing more leads to reps automatically."
                ),
                owner_name="Taylor Morgan",
                calendar_id="sales-team",
            )
        ]
        self._leads = {lead.lead_id: lead for lead in seed}

    async def get_lead(self, lead_id: str) -> LeadRecord:
        lead = self._leads.get(lead_id)
        if lead is None:
            raise KeyError(f"Unknown lead_id: {lead_id}")
        return lead

    async def update_lead_status(self, lead_id: str, *, lifecycle_stage: str) -> LeadRecord:
        lead = await self.get_lead(lead_id)
        updated = lead.model_copy(update={"lifecycle_stage": lifecycle_stage})
        self._leads[lead_id] = updated
        return updated


class InMemoryCallLogRepository:
    def __init__(self):
        self._logs: dict[str, CallLogRecord] = {}

    async def save_call_log(self, call_log: CallLogRecord) -> CallLogRecord:
        self._logs[call_log.conversation_id] = call_log
        return call_log

    async def get_call_log(self, conversation_id: str) -> CallLogRecord | None:
        return self._logs.get(conversation_id)


class InMemorySessionStore:
    def __init__(self):
        self._sessions: dict[str, dict] = {}

    async def get(self, conversation_id: str) -> dict | None:
        state = self._sessions.get(conversation_id)
        if state is None:
            return None
        return copy.deepcopy(state)

    async def save(self, conversation_id: str, state: dict) -> None:
        self._sessions[conversation_id] = copy.deepcopy(state)

    async def delete(self, conversation_id: str) -> None:
        self._sessions.pop(conversation_id, None)
