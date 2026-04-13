from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from ai_sdr_agent.auth.dependencies import get_current_user_id
from ai_sdr_agent.db.engine import get_async_session
from ai_sdr_agent.db.repositories import (
    PgBotConfigRepository,
    PgCallLogRepository,
    PgLeadRepository,
    PgSessionStore,
)
from ai_sdr_agent.graph.service import SDRConversationService, SDRRuntimeDependencies
from ai_sdr_agent.services.brain import build_conversation_brain
from ai_sdr_agent.services.latency_analytics import shared_latency_analytics
from ai_sdr_agent.services.pre_call_loader import PreCallLoader
from ai_sdr_agent.tools import StubCRMGateway, StubCalendarGateway, StubEmailGateway

router = APIRouter(prefix="/api/bots/{bot_id}/test-session", tags=["test"])


class StartTestRequest(BaseModel):
    lead_id: str


class TurnRequest(BaseModel):
    human_input: str


async def _verify_bot(bot_id: str, user_id: uuid.UUID, session: AsyncSession):
    repo = PgBotConfigRepository(session)
    bot = await repo.get(uuid.UUID(bot_id))
    if bot is None or bot.user_id != user_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Bot not found")
    return bot


def _build_service_for_bot(bot_config: dict, lead_repo, session_store, call_log_repo):
    """Build a per-bot SDRConversationService using the bot's config."""
    brain = build_conversation_brain(bot_config=bot_config)
    calendar_gateway = StubCalendarGateway()
    email_gateway = StubEmailGateway()
    crm_gateway = StubCRMGateway()
    pre_call_loader = PreCallLoader(
        lead_repository=lead_repo,
        calendar_gateway=calendar_gateway,
    )
    return SDRConversationService(
        SDRRuntimeDependencies(
            brain=brain,
            calendar_gateway=calendar_gateway,
            email_gateway=email_gateway,
            crm_gateway=crm_gateway,
            pre_call_loader=pre_call_loader,
            session_store=session_store,
            call_log_repository=call_log_repo,
            email_template_path=Path("templates/follow_up_email.html"),
            sales_rep_name=bot_config.get("sales_rep_name", "Sales Team"),
            from_name="AI SDR",
            latency_analytics=shared_latency_analytics,
        ),
        bot_config=bot_config,
    )


@router.post("", status_code=status.HTTP_201_CREATED)
async def start_test_session(
    bot_id: str,
    body: StartTestRequest,
    user_id: uuid.UUID = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_async_session),
):
    bot = await _verify_bot(bot_id, user_id, session)
    bot_cfg = bot.to_config_dict()

    bid = uuid.UUID(bot_id)
    lead_repo = PgLeadRepository(session)
    session_store = PgSessionStore(session, bid)
    call_log_repo = PgCallLogRepository(session, bid)

    svc = _build_service_for_bot(bot_cfg, lead_repo, session_store, call_log_repo)
    conversation_id = await svc.start_session(body.lead_id, bot_config=bot_cfg)
    state = await svc.handle_turn(conversation_id, "")
    await session.commit()
    return {
        "conversation_id": conversation_id,
        "agent_response": state["last_agent_response"],
        "stage": state["current_node"],
    }


@router.post("/{session_id}/turns")
async def run_test_turn(
    bot_id: str,
    session_id: str,
    body: TurnRequest,
    user_id: uuid.UUID = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_async_session),
):
    bot = await _verify_bot(bot_id, user_id, session)
    bot_cfg = bot.to_config_dict()

    bid = uuid.UUID(bot_id)
    lead_repo = PgLeadRepository(session)
    session_store = PgSessionStore(session, bid)
    call_log_repo = PgCallLogRepository(session, bid)

    svc = _build_service_for_bot(bot_cfg, lead_repo, session_store, call_log_repo)
    try:
        state = await svc.handle_turn(session_id, body.human_input)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    await session.commit()
    return {
        "conversation_id": session_id,
        "agent_response": state["last_agent_response"],
        "stage": state["current_node"],
        "call_outcome": state["call_outcome"],
    }
