from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

from ai_sdr_agent.auth.dependencies import get_current_user_id
from ai_sdr_agent.db.engine import get_async_session
from ai_sdr_agent.db.models import BotConfigRow, CallLogRow, LeadRow
from ai_sdr_agent.db.repositories import (
    PgBotConfigRepository,
    PgCallLogRepository,
    PgLeadRepository,
)
from ai_sdr_agent.graph.spec import parse_conversation_spec
from ai_sdr_agent.models import LeadRecord

router = APIRouter(prefix="/api/bots", tags=["bots"])

_SECRET_FIELDS = {
    "openai_api_key",
    "anthropic_api_key",
    "groq_api_key",
    "elevenlabs_api_key",
    "deepgram_api_key",
    "twilio_account_sid",
    "twilio_auth_token",
}

_BOT_UPDATE_FIELDS = {
    "name",
    "is_active",
    "llm_provider",
    "llm_model_name",
    "llm_temperature",
    "llm_max_tokens",
    "openai_api_key",
    "anthropic_api_key",
    "groq_api_key",
    "elevenlabs_api_key",
    "elevenlabs_voice_id",
    "elevenlabs_model_id",
    "deepgram_api_key",
    "deepgram_model",
    "deepgram_language",
    "twilio_account_sid",
    "twilio_auth_token",
    "twilio_phone_number",
    "max_call_turns",
    "max_objection_attempts",
    "max_qualify_attempts",
    "max_booking_attempts",
    "sales_rep_name",
    "prompt_greeting",
    "prompt_qualify",
    "prompt_pitch",
    "prompt_objection",
    "prompt_booking",
    "prompt_wrapup",
    "conversation_spec",
}


class BotCreateRequest(BaseModel):
    name: str = Field(default="New Agent", min_length=1, max_length=200)


class BotUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=200)
    is_active: bool | None = None
    llm_provider: str | None = None
    llm_model_name: str | None = None
    llm_temperature: float | None = None
    llm_max_tokens: int | None = None
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    groq_api_key: str | None = None
    elevenlabs_api_key: str | None = None
    elevenlabs_voice_id: str | None = None
    elevenlabs_model_id: str | None = None
    deepgram_api_key: str | None = None
    deepgram_model: str | None = None
    deepgram_language: str | None = None
    twilio_account_sid: str | None = None
    twilio_auth_token: str | None = None
    twilio_phone_number: str | None = None
    max_call_turns: int | None = None
    max_objection_attempts: int | None = None
    max_qualify_attempts: int | None = None
    max_booking_attempts: int | None = None
    sales_rep_name: str | None = None
    prompt_greeting: str | None = None
    prompt_qualify: str | None = None
    prompt_pitch: str | None = None
    prompt_objection: str | None = None
    prompt_booking: str | None = None
    prompt_wrapup: str | None = None
    conversation_spec: dict[str, Any] | None = None


class LeadCreateRequest(BaseModel):
    lead_name: str = Field(..., min_length=1, max_length=200)
    company: str = ""
    phone_number: str = Field(..., min_length=1, max_length=30)
    lead_email: str = ""
    lead_context: str = ""
    lifecycle_stage: str = "follow_up"
    timezone: str = "UTC"
    owner_name: str = "Sales Team"
    calendar_id: str = "sales-team"
    metadata: dict[str, Any] = Field(default_factory=dict)


def _mask_secret(value: str | None) -> str | None:
    if not value:
        return None
    return f"****{value[-4:]}" if len(value) > 4 else "****"


def _public_bot(row: BotConfigRow) -> dict[str, Any]:
    data = {
        "id": str(row.id),
        "name": row.name,
        "is_active": row.is_active,
        "llm_provider": row.llm_provider,
        "llm_model_name": row.llm_model_name,
        "llm_temperature": row.llm_temperature,
        "llm_max_tokens": row.llm_max_tokens,
        "openai_api_key": _mask_secret(row.openai_api_key),
        "anthropic_api_key": _mask_secret(row.anthropic_api_key),
        "groq_api_key": _mask_secret(row.groq_api_key),
        "elevenlabs_api_key": _mask_secret(row.elevenlabs_api_key),
        "elevenlabs_voice_id": row.elevenlabs_voice_id,
        "elevenlabs_model_id": row.elevenlabs_model_id,
        "deepgram_api_key": _mask_secret(row.deepgram_api_key),
        "deepgram_model": row.deepgram_model,
        "deepgram_language": row.deepgram_language,
        "twilio_account_sid": _mask_secret(row.twilio_account_sid),
        "twilio_auth_token": _mask_secret(row.twilio_auth_token),
        "twilio_phone_number": row.twilio_phone_number,
        "max_call_turns": row.max_call_turns,
        "max_objection_attempts": row.max_objection_attempts,
        "max_qualify_attempts": row.max_qualify_attempts,
        "max_booking_attempts": row.max_booking_attempts,
        "sales_rep_name": row.sales_rep_name,
        "prompt_greeting": row.prompt_greeting,
        "prompt_qualify": row.prompt_qualify,
        "prompt_pitch": row.prompt_pitch,
        "prompt_objection": row.prompt_objection,
        "prompt_booking": row.prompt_booking,
        "prompt_wrapup": row.prompt_wrapup,
        "conversation_spec": row.conversation_spec,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }
    return data


def _public_lead(row: LeadRow) -> dict[str, Any]:
    return {
        "id": str(row.id),
        "bot_id": str(row.bot_id),
        "lead_name": row.lead_name,
        "company": row.company,
        "phone_number": row.phone_number,
        "lead_email": row.lead_email,
        "lead_context": row.lead_context,
        "lifecycle_stage": row.lifecycle_stage,
        "timezone": row.timezone,
        "owner_name": row.owner_name,
        "calendar_id": row.calendar_id,
        "metadata": row.metadata_json or {},
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


def _public_call(row: CallLogRow) -> dict[str, Any]:
    return {
        "id": str(row.id),
        "bot_id": str(row.bot_id),
        "conversation_id": row.conversation_id,
        "lead_id": row.lead_id,
        "started_at": row.started_at.isoformat() if row.started_at else None,
        "completed_at": row.completed_at.isoformat() if row.completed_at else None,
        "call_outcome": row.call_outcome,
        "transcript": row.transcript or [],
        "qualification_notes": row.qualification_notes or {},
        "meeting_booked": row.meeting_booked,
        "proposed_slot": row.proposed_slot,
        "follow_up_action": row.follow_up_action,
    }


async def _verify_bot(bot_id: str, user_id: uuid.UUID, session: AsyncSession) -> BotConfigRow:
    try:
        parsed_bot_id = uuid.UUID(bot_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Bot not found") from exc

    repo = PgBotConfigRepository(session)
    bot = await repo.get(parsed_bot_id)
    if bot is None or bot.user_id != user_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Bot not found")
    return bot


@router.get("")
async def list_bots(
    user_id: uuid.UUID = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_async_session),
):
    repo = PgBotConfigRepository(session)
    rows = await repo.list_for_user(user_id)
    return [_public_bot(row) for row in rows]


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_bot(
    body: BotCreateRequest,
    user_id: uuid.UUID = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_async_session),
):
    repo = PgBotConfigRepository(session)
    row = await repo.create(user_id=user_id, name=body.name)
    await session.commit()
    return _public_bot(row)


@router.get("/{bot_id}")
async def get_bot(
    bot_id: str,
    user_id: uuid.UUID = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_async_session),
):
    bot = await _verify_bot(bot_id, user_id, session)
    return _public_bot(bot)


@router.patch("/{bot_id}")
async def update_bot(
    bot_id: str,
    body: BotUpdateRequest,
    user_id: uuid.UUID = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_async_session),
):
    bot = await _verify_bot(bot_id, user_id, session)
    fields = body.model_dump(exclude_unset=True)
    fields = {k: v for k, v in fields.items() if k in _BOT_UPDATE_FIELDS}
    for key in list(fields):
        if key in _SECRET_FIELDS and isinstance(fields[key], str) and "****" in fields[key]:
            fields.pop(key)
    if "conversation_spec" in fields and fields["conversation_spec"] is not None:
        try:
            parse_conversation_spec(fields["conversation_spec"])
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    repo = PgBotConfigRepository(session)
    row = await repo.update(bot.id, **fields)
    await session.commit()
    return _public_bot(row or bot)


@router.delete("/{bot_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_bot(
    bot_id: str,
    user_id: uuid.UUID = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_async_session),
):
    bot = await _verify_bot(bot_id, user_id, session)
    repo = PgBotConfigRepository(session)
    await repo.delete(bot.id)
    await session.commit()


@router.get("/{bot_id}/leads")
async def list_leads(
    bot_id: str,
    user_id: uuid.UUID = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_async_session),
):
    bot = await _verify_bot(bot_id, user_id, session)
    repo = PgLeadRepository(session)
    rows = await repo.list_for_bot(bot.id)
    return [_public_lead(row) for row in rows]


@router.post("/{bot_id}/leads", status_code=status.HTTP_201_CREATED)
async def create_lead(
    bot_id: str,
    body: LeadCreateRequest,
    user_id: uuid.UUID = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_async_session),
):
    bot = await _verify_bot(bot_id, user_id, session)
    repo = PgLeadRepository(session)
    row = await repo.create_lead(
        bot.id,
        LeadRecord(
            lead_id="",
            lead_name=body.lead_name,
            company=body.company,
            phone_number=body.phone_number,
            lead_email=body.lead_email,
            lead_context=body.lead_context,
            lifecycle_stage=body.lifecycle_stage,
            timezone=body.timezone,
            owner_name=body.owner_name,
            calendar_id=body.calendar_id,
            metadata=body.metadata,
        ),
    )
    await session.commit()
    return _public_lead(row)


@router.get("/{bot_id}/calls")
async def list_calls(
    bot_id: str,
    user_id: uuid.UUID = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_async_session),
):
    bot = await _verify_bot(bot_id, user_id, session)
    repo = PgCallLogRepository(session, bot.id)
    rows = await repo.list_for_bot(bot.id)
    return [_public_call(row) for row in rows]
