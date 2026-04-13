from unittest.mock import AsyncMock

import pytest

from ai_sdr_agent.graph.routers import (
    apply_qualify_router_safety,
    route_after_objection,
    route_after_pitch,
    route_after_qualify,
)
from ai_sdr_agent.graph.state import build_initial_state
from ai_sdr_agent.services.brain import StubConversationBrain


def make_state(last_human_message: str):
    state = build_initial_state(
        lead_id="lead-001",
        lead_name="Jordan Lee",
        lead_email="jordan.lee@example.com",
        phone_number="+15551234567",
        company="Northwind Logistics",
        calendar_id="sales-team",
        lead_context="Requested a follow-up about lead response times.",
        available_slots=[],
    )
    state["transcript"].append({"role": "human", "content": last_human_message})
    state["last_human_message"] = last_human_message
    return state


@pytest.mark.asyncio
async def test_route_after_qualify_respects_pitch_from_brain():
    brain = AsyncMock()
    brain.classify = AsyncMock(return_value="pitch")
    decision = await route_after_qualify(make_state("VP of sales with budget this quarter."), brain)
    assert decision == "pitch"


@pytest.mark.asyncio
async def test_route_after_qualify_safety_downgrades_false_not_interested():
    brain = AsyncMock()
    brain.classify = AsyncMock(return_value="not_interested")
    decision = await route_after_qualify(make_state("yeah yeah don't"), brain)
    assert decision == "continue_qualifying"


@pytest.mark.asyncio
async def test_route_after_pitch_to_objection():
    decision = await route_after_pitch(make_state("We already have something in place."), StubConversationBrain())
    assert decision == "handle_objection"


@pytest.mark.asyncio
async def test_route_after_objection_to_wrap_up():
    decision = await route_after_objection(make_state("Not interested, please remove me."), StubConversationBrain())
    assert decision == "wrap_up"


def test_apply_qualify_router_safety_overrides_garbled_yeah():
    assert apply_qualify_router_safety("yeah yeah don't", "not_interested") == "continue_qualifying"


def test_apply_qualify_router_safety_keeps_explicit_refusal():
    assert apply_qualify_router_safety("Look, we're not interested.", "not_interested") == "not_interested"


def test_apply_qualify_router_safety_passes_through_other_labels():
    assert apply_qualify_router_safety("yeah", "pitch") == "pitch"
