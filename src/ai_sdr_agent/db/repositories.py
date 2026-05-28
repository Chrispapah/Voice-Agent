from __future__ import annotations

import copy
import uuid
from typing import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ai_sdr_agent.db.models import (
    AgentToolRow,
    AuthConnectionRow,
    BotConfigRow,
    CallLogRow,
    LeadRow,
    SessionRow,
    WorkspaceEnvVarRow,
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
                completed_at=call_log.completed_at,
                call_outcome=call_log.call_outcome,
                call_quality=call_log.call_quality,
                transcript=call_log.transcript,
                qualification_notes=call_log.qualification_notes,
                meeting_booked=call_log.meeting_booked,
                proposed_slot=call_log.proposed_slot,
                follow_up_action=call_log.follow_up_action,
            )
            self.session.add(row)
        else:
            row.completed_at = call_log.completed_at
            row.call_outcome = call_log.call_outcome
            row.call_quality = call_log.call_quality
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
            call_quality=row.call_quality or "needs_attention",
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
        if row.bot_id != self.bot_id:
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
        if row is not None and row.bot_id == self.bot_id:
            await self.session.delete(row)
            await self.session.flush()


# ---------------------------------------------------------------------------
# Agent tools
# ---------------------------------------------------------------------------

class PgAgentToolRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def list_by_ids(
        self,
        *,
        user_id: uuid.UUID,
        tool_ids: Sequence[uuid.UUID | str],
    ) -> Sequence[AgentToolRow]:
        if not tool_ids:
            return []
        ids = [uuid.UUID(str(tid)) for tid in tool_ids]
        result = await self.session.execute(
            select(AgentToolRow).where(
                AgentToolRow.user_id == user_id,
                AgentToolRow.id.in_(ids),
                AgentToolRow.is_active.is_(True),
            )
        )
        return result.scalars().all()

    async def get(self, tool_id: uuid.UUID, user_id: uuid.UUID) -> AgentToolRow | None:
        result = await self.session.execute(
            select(AgentToolRow).where(
                AgentToolRow.id == tool_id,
                AgentToolRow.user_id == user_id,
            )
        )
        return result.scalar_one_or_none()


# ---------------------------------------------------------------------------
# Workspace env vars
# ---------------------------------------------------------------------------

class PgWorkspaceEnvVarRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def list_for_user(self, user_id: uuid.UUID) -> Sequence[WorkspaceEnvVarRow]:
        result = await self.session.execute(
            select(WorkspaceEnvVarRow)
            .where(WorkspaceEnvVarRow.user_id == user_id)
            .order_by(WorkspaceEnvVarRow.name.asc())
        )
        return result.scalars().all()

    async def get_by_name(self, user_id: uuid.UUID, name: str) -> WorkspaceEnvVarRow | None:
        result = await self.session.execute(
            select(WorkspaceEnvVarRow).where(
                WorkspaceEnvVarRow.user_id == user_id,
                WorkspaceEnvVarRow.name == name,
            )
        )
        return result.scalar_one_or_none()

    async def create(
        self,
        *,
        user_id: uuid.UUID,
        name: str,
        value: str,
    ) -> WorkspaceEnvVarRow:
        row = WorkspaceEnvVarRow(user_id=user_id, name=name, value=value)
        self.session.add(row)
        await self.session.flush()
        return row

    async def update(
        self,
        var_id: uuid.UUID,
        user_id: uuid.UUID,
        **fields: str,
    ) -> WorkspaceEnvVarRow | None:
        row = await self.session.get(WorkspaceEnvVarRow, var_id)
        if row is None or row.user_id != user_id:
            return None
        for key, value in fields.items():
            if hasattr(row, key):
                setattr(row, key, value)
        await self.session.flush()
        return row

    async def delete(self, var_id: uuid.UUID, user_id: uuid.UUID) -> bool:
        row = await self.session.get(WorkspaceEnvVarRow, var_id)
        if row is None or row.user_id != user_id:
            return False
        await self.session.delete(row)
        await self.session.flush()
        return True


# ---------------------------------------------------------------------------
# Auth connections
# ---------------------------------------------------------------------------

class PgAuthConnectionRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def list_for_user(self, user_id: uuid.UUID) -> Sequence[AuthConnectionRow]:
        result = await self.session.execute(
            select(AuthConnectionRow)
            .where(AuthConnectionRow.user_id == user_id)
            .order_by(AuthConnectionRow.label.asc())
        )
        return result.scalars().all()

    async def get(self, connection_id: uuid.UUID, user_id: uuid.UUID) -> AuthConnectionRow | None:
        row = await self.session.get(AuthConnectionRow, connection_id)
        if row is None or row.user_id != user_id:
            return None
        return row

    async def create(
        self,
        *,
        user_id: uuid.UUID,
        label: str,
        type: str,
        config_json: dict,
    ) -> AuthConnectionRow:
        row = AuthConnectionRow(
            user_id=user_id,
            label=label,
            type=type,
            config_json=config_json,
        )
        self.session.add(row)
        await self.session.flush()
        return row

    async def delete(self, connection_id: uuid.UUID, user_id: uuid.UUID) -> bool:
        row = await self.get(connection_id, user_id)
        if row is None:
            return False
        await self.session.delete(row)
        await self.session.flush()
        return True
