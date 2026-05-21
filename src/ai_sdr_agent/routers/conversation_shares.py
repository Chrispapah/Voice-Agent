from __future__ import annotations

import hashlib
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ai_sdr_agent.auth.dependencies import get_current_user_id
from ai_sdr_agent.db.engine import get_async_session
from ai_sdr_agent.db.models import BotConfigRow, CallLogRow, ConversationShareRow

router = APIRouter(tags=["conversation-shares"])


class ConversationShareCreateRequest(BaseModel):
    expires_in_days: int | None = Field(default=30, ge=1, le=365)


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _safe_transcript(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    turns: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            continue
        turns.append({"role": role, "content": content})
    return turns


def _share_response(row: ConversationShareRow, token: str, request: Request) -> dict[str, Any]:
    preview_path = f"/preview/conversation/{token}"
    origin = request.headers.get("origin") or str(request.base_url).rstrip("/")
    return {
        "id": str(row.id),
        "call_log_id": str(row.call_log_id),
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "expires_at": row.expires_at.isoformat() if row.expires_at else None,
        "revoked_at": row.revoked_at.isoformat() if row.revoked_at else None,
        "token": token,
        "preview_path": preview_path,
        "preview_url": f"{origin}{preview_path}",
    }


def _public_preview(row: ConversationShareRow, call_log: CallLogRow, bot: BotConfigRow) -> dict[str, Any]:
    return {
        "share": {
            "id": str(row.id),
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "expires_at": row.expires_at.isoformat() if row.expires_at else None,
        },
        "conversation": {
            "conversation_id": call_log.conversation_id,
            "agent_name": bot.name,
            "started_at": call_log.started_at.isoformat() if call_log.started_at else None,
            "completed_at": call_log.completed_at.isoformat() if call_log.completed_at else None,
            "call_outcome": call_log.call_outcome,
            "transcript": _safe_transcript(call_log.transcript),
            "meeting_booked": call_log.meeting_booked,
            "proposed_slot": call_log.proposed_slot,
            "follow_up_action": call_log.follow_up_action,
        },
    }


async def _get_owned_call_log(
    call_log_id: str,
    user_id: uuid.UUID,
    session: AsyncSession,
) -> CallLogRow:
    try:
        parsed_call_log_id = uuid.UUID(call_log_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Call log not found") from exc

    result = await session.execute(
        select(CallLogRow)
        .join(BotConfigRow, CallLogRow.bot_id == BotConfigRow.id)
        .where(CallLogRow.id == parsed_call_log_id, BotConfigRow.user_id == user_id)
    )
    row = result.scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Call log not found")
    return row


@router.post("/api/call-logs/{call_log_id}/share", status_code=status.HTTP_201_CREATED)
async def create_conversation_share(
    call_log_id: str,
    body: ConversationShareCreateRequest,
    request: Request,
    user_id: uuid.UUID = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_async_session),
):
    call_log = await _get_owned_call_log(call_log_id, user_id, session)
    token = secrets.token_urlsafe(32)
    expires_at = _utcnow() + timedelta(days=body.expires_in_days or 30)
    row = ConversationShareRow(
        call_log_id=call_log.id,
        token_hash=_token_hash(token),
        created_by=user_id,
        expires_at=expires_at,
    )
    session.add(row)
    await session.commit()
    return _share_response(row, token, request)


@router.post("/api/conversation-shares/{share_id}/revoke")
async def revoke_conversation_share(
    share_id: str,
    user_id: uuid.UUID = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_async_session),
):
    try:
        parsed_share_id = uuid.UUID(share_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Share not found") from exc

    result = await session.execute(
        select(ConversationShareRow)
        .join(CallLogRow, ConversationShareRow.call_log_id == CallLogRow.id)
        .join(BotConfigRow, CallLogRow.bot_id == BotConfigRow.id)
        .where(ConversationShareRow.id == parsed_share_id, BotConfigRow.user_id == user_id)
    )
    row = result.scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Share not found")

    row.revoked_at = _utcnow()
    await session.commit()
    return {"id": str(row.id), "revoked_at": row.revoked_at.isoformat()}


@router.get("/api/public/conversation-previews/{token}")
async def get_public_conversation_preview(
    token: str,
    session: AsyncSession = Depends(get_async_session),
):
    result = await session.execute(
        select(ConversationShareRow, CallLogRow, BotConfigRow)
        .join(CallLogRow, ConversationShareRow.call_log_id == CallLogRow.id)
        .join(BotConfigRow, CallLogRow.bot_id == BotConfigRow.id)
        .where(ConversationShareRow.token_hash == _token_hash(token))
    )
    row = result.one_or_none()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Preview link not found")

    share, call_log, bot = row
    now = _utcnow()
    if share.revoked_at is not None or (share.expires_at is not None and share.expires_at <= now):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Preview link not found")

    return _public_preview(share, call_log, bot)
