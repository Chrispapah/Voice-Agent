from __future__ import annotations

import copy
import uuid
from typing import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ai_sdr_agent.db.models import (
    BotConfigRow,
    CallLogRow,
    LeadRow,
    SessionRow,
)
from ai_sdr_agent.models import CallLogRecord, LeadRecord


# ---------------------------------------------------------------------------
# BotConfig repository
# ---------------------------------------------------------------------------

class PgBotConfigRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, *, user_id: uuid.UUID, name: str = "My Bot", **overrides) -> BotConfigRow:
        row = BotConfigRow(user_id=user_id, name=name, **overrides)
        self.session.add(row)
        await self.session.flush()
        return row

    async def get(self, bot_id: uuid.UUID) -> BotConfigRow | None:
        return await self.session.get(BotConfigRow, bot_id)

    async def list_for_user(self, user_id: uuid.UUID) -> Sequence[BotConfigRow]:
        result = await self.session.execute(
            select(BotConfigRow).where(BotConfigRow.user_id == user_id).order_by(BotConfigRow.created_at.desc())
        )
        return result.scalars().all()

    async def update(self, bot_id: uuid.UUID, **fields) -> BotConfigRow | None:
        row = await self.get(bot_id)
        if row is None:
            return None
        for key, value in fields.items():
            if hasattr(row, key):
                setattr(row, key, value)
        await self.session.flush()
        return row

    async def delete(self, bot_id: uuid.UUID) -> bool:
        row = await self.get(bot_id)
        if row is None:
            return False
        await self.session.delete(row)
        await self.session.flush()
        return True


# ---------------------------------------------------------------------------
# Lead repository  (satisfies the LeadRepository protocol)
# ---------------------------------------------------------------------------

class PgLeadRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_lead(self, bot_id: uuid.UUID, lead: LeadRecord) -> LeadRow:
        row = LeadRow(
            bot_id=bot_id,
            lead_name=lead.lead_name,
            company=lead.company,
            phone_number=lead.phone_number,
            lead_email=lead.lead_email,
            lead_context=lead.lead_context,
            lifecycle_stage=lead.lifecycle_stage,
            timezone=lead.timezone,
            owner_name=lead.owner_name,
            calendar_id=lead.calendar_id,
            metadata_json=lead.metadata,
        )
        self.session.add(row)
        await self.session.flush()
        return row

    async def list_for_bot(self, bot_id: uuid.UUID) -> Sequence[LeadRow]:
        result = await self.session.execute(
            select(LeadRow).where(LeadRow.bot_id == bot_id).order_by(LeadRow.created_at.desc())
        )
        return result.scalars().all()

    def _row_to_record(self, row: LeadRow) -> LeadRecord:
        return LeadRecord(
            lead_id=str(row.id),
            lead_name=row.lead_name,
            company=row.company,
            phone_number=row.phone_number,
            lead_email=row.lead_email,
            lead_context=row.lead_context,
            lifecycle_stage=row.lifecycle_stage,
            timezone=row.timezone,
            owner_name=row.owner_name,
            calendar_id=row.calendar_id,
            metadata=row.metadata_json or {},
        )

    async def get_lead(self, lead_id: str) -> LeadRecord:
        row = await self.session.get(LeadRow, uuid.UUID(lead_id))
        if row is None:
            raise KeyError(f"Unknown lead_id: {lead_id}")
        return self._row_to_record(row)

    async def update_lead_status(self, lead_id: str, *, lifecycle_stage: str) -> LeadRecord:
        row = await self.session.get(LeadRow, uuid.UUID(lead_id))
        if row is None:
            raise KeyError(f"Unknown lead_id: {lead_id}")
        row.lifecycle_stage = lifecycle_stage
        await self.session.flush()
        return self._row_to_record(row)


# ---------------------------------------------------------------------------
# Call log repository (satisfies the CallLogRepository protocol)
# ---------------------------------------------------------------------------

class PgCallLogRepository:
    def __init__(self, session: AsyncSession, bot_id: uuid.UUID):
        self.session = session
        self.bot_id = bot_id

    async def save_call_log(self, call_log: CallLogRecord) -> CallLogRecord:
        existing = await self.session.execute(
            select(CallLogRow).where(CallLogRow.conversation_id == call_log.conversation_id)
        )
        row = existing.scalar_one_or_none()
        if row is None:
            row = CallLogRow(
                bot_id=self.bot_id,
                conversation_id=call_log.conversation_id,
                lead_id=call_log.lead_id,
                call_outcome=call_log.call_outcome,
                transcript=call_log.transcript,
                qualification_notes=call_log.qualification_notes,
                meeting_booked=call_log.meeting_booked,
                proposed_slot=call_log.proposed_slot,
                follow_up_action=call_log.follow_up_action,
            )
            self.session.add(row)
        else:
            row.call_outcome = call_log.call_outcome
            row.transcript = call_log.transcript
            row.qualification_notes = call_log.qualification_notes
            row.meeting_booked = call_log.meeting_booked
            row.proposed_slot = call_log.proposed_slot
            row.follow_up_action = call_log.follow_up_action
        await self.session.flush()
        return call_log

    async def get_call_log(self, conversation_id: str) -> CallLogRecord | None:
        result = await self.session.execute(
            select(CallLogRow).where(CallLogRow.conversation_id == conversation_id)
        )
        row = result.scalar_one_or_none()
        if row is None:
            return None
        return CallLogRecord(
            conversation_id=row.conversation_id,
            lead_id=row.lead_id,
            started_at=row.started_at,
            completed_at=row.completed_at,
            call_outcome=row.call_outcome,
            transcript=row.transcript or [],
            qualification_notes=row.qualification_notes or {},
            meeting_booked=row.meeting_booked,
            proposed_slot=row.proposed_slot,
            follow_up_action=row.follow_up_action,
        )

    async def list_for_bot(self, bot_id: uuid.UUID | None = None) -> Sequence[CallLogRow]:
        bid = bot_id or self.bot_id
        result = await self.session.execute(
            select(CallLogRow).where(CallLogRow.bot_id == bid).order_by(CallLogRow.started_at.desc())
        )
        return result.scalars().all()


# ---------------------------------------------------------------------------
# Session store (satisfies the SessionStore protocol)
# ---------------------------------------------------------------------------

class PgSessionStore:
    def __init__(self, session: AsyncSession, bot_id: uuid.UUID):
        self.session = session
        self.bot_id = bot_id

    async def get(self, conversation_id: str) -> dict | None:
        row = await self.session.get(SessionRow, conversation_id)
        if row is None:
            return None
        return copy.deepcopy(row.state_json)

    async def save(self, conversation_id: str, state: dict) -> None:
        row = await self.session.get(SessionRow, conversation_id)
        if row is None:
            row = SessionRow(
                conversation_id=conversation_id,
                bot_id=self.bot_id,
                state_json=copy.deepcopy(state),
            )
            self.session.add(row)
        else:
            row.state_json = copy.deepcopy(state)
        await self.session.flush()

    async def delete(self, conversation_id: str) -> None:
        row = await self.session.get(SessionRow, conversation_id)
        if row is not None:
            await self.session.delete(row)
            await self.session.flush()
