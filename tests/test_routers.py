import pytest

from ai_sdr_agent.graph.routers import route_after_objection, route_after_pitch, route_after_qualify
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
async def test_route_after_qualify_to_pitch():
    decision = await route_after_qualify(make_state("Yes, I own that process."), StubConversationBrain())
    assert decision == "pitch"


@pytest.mark.asyncio
async def test_route_after_pitch_to_objection():
    decision = await route_after_pitch(make_state("We already have something in place."), StubConversationBrain())
    assert decision == "handle_objection"


@pytest.mark.asyncio
async def test_route_after_objection_to_wrap_up():
    decision = await route_after_objection(make_state("Not interested, please remove me."), StubConversationBrain())
    assert decision == "wrap_up"
