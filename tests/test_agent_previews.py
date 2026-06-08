from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from ai_sdr_agent.db.models import AgentPreviewShareRow, BotConfigRow
from ai_sdr_agent.routers.agent_previews import (
    _preview_lead_id,
    _preview_lead_repo,
    _token_hash,
    start_public_agent_preview_session,
)


class _ShareResult:
    def __init__(self, share: AgentPreviewShareRow, bot: BotConfigRow):
        self._row = (share, bot)

    def one_or_none(self):
        return self._row


class _Session:
    def __init__(self, share: AgentPreviewShareRow, bot: BotConfigRow):
        self._results = [_ShareResult(share, bot)]
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
async def test_start_public_preview_session_uses_synthetic_lead_without_touching_leads_table():
    token = "preview-token"
    share, bot = _preview_share_rows(token)
    session = _Session(share, bot)

    payload = await start_public_agent_preview_session(token, session=session)

    assert payload == {
        "lead_id": f"preview-share-{share.id}",
        "conversation_id": None,
        "voice_provider": "builtin",
    }
    assert share.session_count == 1
    assert session._results == []
    assert session.added == []
    assert session.flushed is False
    assert session.committed is True


@pytest.mark.asyncio
async def test_preview_lead_repo_builds_synthetic_visitor_lead():
    share, _ = _preview_share_rows("preview-token")
    repo = _preview_lead_repo(share)

    lead = await repo.get_lead(_preview_lead_id(share))

    assert lead.lead_id == f"preview-share-{share.id}"
    assert lead.lead_name == "Visitor"
    assert lead.lifecycle_stage == "follow_up"
    assert lead.metadata == {"source": "agent_preview", "share_id": str(share.id)}
