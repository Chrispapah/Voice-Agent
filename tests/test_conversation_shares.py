from __future__ import annotations

import uuid
import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi import HTTPException

from ai_sdr_agent.db.models import BotConfigRow, CallLogRow, ConversationShareRow

_ROUTER_PATH = Path(__file__).resolve().parents[1] / "src" / "ai_sdr_agent" / "routers" / "conversation_shares.py"
_SPEC = importlib.util.spec_from_file_location("conversation_shares_under_test", _ROUTER_PATH)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
get_public_conversation_preview = _MODULE.get_public_conversation_preview


class _Result:
    def __init__(self, row):
        self._row = row

    def one_or_none(self):
        return self._row


class _Session:
    def __init__(self, row):
        self._row = row

    async def execute(self, _query):
        return _Result(self._row)


def _preview_rows(*, revoked: bool = False, expired: bool = False):
    now = datetime.now(timezone.utc)
    bot = BotConfigRow(id=uuid.uuid4(), user_id=uuid.uuid4(), name="Demo Agent")
    call_log = CallLogRow(
        id=uuid.uuid4(),
        bot_id=bot.id,
        conversation_id="conv-123",
        lead_id="lead-123",
        started_at=now,
        completed_at=now + timedelta(minutes=2),
        call_outcome="meeting_booked",
        transcript=[
            {"role": "human", "content": "Can you show me the product?"},
            {"role": "assistant", "content": "Yes, let's book a demo."},
            {"role": "system", "private": "ignored"},
        ],
        qualification_notes={"budget": "private"},
        meeting_booked=True,
        proposed_slot="Tomorrow at 3 PM",
        follow_up_action="send_calendar_invite",
    )
    share = ConversationShareRow(
        id=uuid.uuid4(),
        call_log_id=call_log.id,
        token_hash="hash",
        created_by=bot.user_id,
        created_at=now,
        expires_at=now - timedelta(days=1) if expired else now + timedelta(days=30),
        revoked_at=now if revoked else None,
    )
    return share, call_log, bot


@pytest.mark.asyncio
async def test_public_preview_returns_sanitized_conversation():
    payload = await get_public_conversation_preview("token", session=_Session(_preview_rows()))

    assert payload["conversation"]["conversation_id"] == "conv-123"
    assert payload["conversation"]["agent_name"] == "Demo Agent"
    assert payload["conversation"]["transcript"] == [
        {"role": "human", "content": "Can you show me the product?"},
        {"role": "assistant", "content": "Yes, let's book a demo."},
    ]
    assert "qualification_notes" not in payload["conversation"]


@pytest.mark.asyncio
async def test_public_preview_rejects_revoked_share():
    with pytest.raises(HTTPException) as exc_info:
        await get_public_conversation_preview("token", session=_Session(_preview_rows(revoked=True)))

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_public_preview_rejects_expired_share():
    with pytest.raises(HTTPException) as exc_info:
        await get_public_conversation_preview("token", session=_Session(_preview_rows(expired=True)))

    assert exc_info.value.status_code == 404
