from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from ai_sdr_agent.db.models import AgentPreviewShareRow, BotConfigRow, LeadRow
from ai_sdr_agent.routers.agent_previews import _token_hash, start_public_agent_preview_session


class _ShareResult:
    def __init__(self, share: AgentPreviewShareRow, bot: BotConfigRow):
        self._row = (share, bot)

    def one_or_none(self):
        return self._row


class _LeadResult:
    def __init__(self, lead: LeadRow | None):
        self._lead = lead

    def scalar_one_or_none(self):
        return self._lead


class _Session:
    def __init__(self, share: AgentPreviewShareRow, bot: BotConfigRow, lead: LeadRow | None = None):
        self._results = [_ShareResult(share, bot), _LeadResult(lead)]
        self.added = []
        self.flushed = False
        self.committed = False

    async def execute(self, _query):
        return self._results.pop(0)

    def add(self, row):
        self.added.append(row)

    async def flush(self):
        self.flushed = True

    async def commit(self):
        self.committed = True


def _preview_share_rows(token: str):
    now = datetime.now(timezone.utc)
    bot = BotConfigRow(id=uuid.uuid4(), user_id=uuid.uuid4(), name="Demo Agent", voice_provider="builtin")
    share = AgentPreviewShareRow(
        id=uuid.uuid4(),
        bot_id=bot.id,
        token_hash=_token_hash(token),
        created_by=bot.user_id,
        created_at=now,
        expires_at=now + timedelta(days=30),
        max_sessions=100,
        session_count=0,
    )
    return share, bot


@pytest.mark.asyncio
async def test_start_public_preview_session_reuses_existing_preview_lead():
    token = "preview-token"
    share, bot = _preview_share_rows(token)
    lead = LeadRow(
        id=uuid.uuid4(),
        bot_id=bot.id,
        lead_name="Visitor",
        company="",
        phone_number=f"preview-share-{share.id}",
        lead_email="",
        lead_context="",
        lifecycle_stage="follow_up",
        timezone="UTC",
        owner_name="",
        calendar_id="",
        metadata_json={"source": "agent_preview", "share_id": str(share.id)},
    )
    session = _Session(share, bot, lead)

    payload = await start_public_agent_preview_session(token, session=session)

    assert payload == {
        "lead_id": str(lead.id),
        "conversation_id": None,
        "voice_provider": "builtin",
    }
    assert share.session_count == 1
    assert session.added == []
    assert session.committed is True


@pytest.mark.asyncio
async def test_start_public_preview_session_creates_preview_lead_with_standard_lifecycle():
    token = "preview-token"
    share, bot = _preview_share_rows(token)
    session = _Session(share, bot, None)

    payload = await start_public_agent_preview_session(token, session=session)

    assert payload["conversation_id"] is None
    assert payload["voice_provider"] == "builtin"
    assert share.session_count == 1
    assert len(session.added) == 1
    assert session.added[0].lifecycle_stage == "follow_up"
    assert session.added[0].metadata_json == {"source": "agent_preview", "share_id": str(share.id)}
    assert session.flushed is True
    assert session.committed is True
