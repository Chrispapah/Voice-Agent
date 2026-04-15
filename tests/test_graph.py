from pathlib import Path
import asyncio

import pytest

from ai_sdr_agent.graph.service import SDRConversationService, SDRRuntimeDependencies
from ai_sdr_agent.services.brain import StubConversationBrain
from ai_sdr_agent.services.persistence import (
    InMemoryCallLogRepository,
    InMemoryLeadRepository,
    InMemorySessionStore,
)
from ai_sdr_agent.services.pre_call_loader import PreCallLoader
from ai_sdr_agent.tools import StubCRMGateway, StubCalendarGateway, StubEmailGateway


@pytest.fixture
def conversation_service():
    lead_repository = InMemoryLeadRepository()
    seed_lead = lead_repository._leads["lead-001"]
    calendar_gateway = StubCalendarGateway()
    email_gateway = StubEmailGateway()
    crm_gateway = StubCRMGateway(seed_leads=[seed_lead])
    service = SDRConversationService(
        SDRRuntimeDependencies(
            brain=StubConversationBrain(),
            calendar_gateway=calendar_gateway,
            email_gateway=email_gateway,
            crm_gateway=crm_gateway,
            pre_call_loader=PreCallLoader(
                lead_repository=lead_repository,
                calendar_gateway=calendar_gateway,
            ),
            session_store=InMemorySessionStore(),
            call_log_repository=InMemoryCallLogRepository(),
            email_template_path=Path("templates/follow_up_email.html"),
            sales_rep_name="Taylor Morgan",
            from_name="AI SDR",
        )
    )
    return service, calendar_gateway, email_gateway, crm_gateway


@pytest.mark.asyncio
async def test_graph_advances_one_stage_per_turn(conversation_service):
    service, _, _, _ = conversation_service
    conversation_id = await service.start_session("lead-001")

    state = await service.handle_turn(conversation_id, "")
    assert state["current_node"] == "greeting"
    assert state["next_node"] == "qualify_lead"

    state = await service.handle_turn(conversation_id, "Yes, I oversee sales operations.")
    assert state["current_node"] == "qualify_lead"
    assert state["next_node"] == "pitch"

    state = await service.handle_turn(conversation_id, "Yes, sounds interesting.")
    assert state["current_node"] == "pitch"
    assert state["next_node"] == "book_meeting"


@pytest.mark.asyncio
async def test_phone_session_starts_after_seeded_greeting(conversation_service):
    service, _, _, _ = conversation_service
    opening = "Hi, this is John. Do you have 30 seconds?"
    conversation_id = await service.start_session(
        "lead-001",
        initial_agent_message=opening,
        initial_current_node="greeting",
        initial_next_node="qualify_lead",
    )

    state = await service.get_state(conversation_id)
    assert state["current_node"] == "greeting"
    assert state["next_node"] == "qualify_lead"
    assert state["last_agent_response"] == opening
    assert state["transcript"] == [{"role": "agent", "content": opening}]


@pytest.mark.asyncio
async def test_booking_path_updates_state_and_side_effects(conversation_service):
    service, calendar_gateway, email_gateway, crm_gateway = conversation_service
    conversation_id = await service.start_session("lead-001")

    await service.handle_turn(conversation_id, "")
    await service.handle_turn(conversation_id, "Yes, I run sales operations.")
    await service.handle_turn(conversation_id, "Yes, let's do it.")
    state = await service.handle_turn(conversation_id, "Tomorrow at 3 PM works.")

    assert state["current_node"] == "book_meeting"
    assert state["meeting_booked"] is True
    assert state["next_node"] == "wrap_up"
    assert len(calendar_gateway.bookings) == 1

    state = await service.handle_turn(conversation_id, "Booked sounds good.")
    assert state["current_node"] == "wrap_up"
    assert state["call_outcome"] == "meeting_booked"
    assert len(email_gateway.sent_messages) == 1
    assert len(crm_gateway.updates) == 1


@pytest.mark.asyncio
async def test_objection_loops_back_to_pitch(conversation_service):
    service, _, _, _ = conversation_service
    conversation_id = await service.start_session("lead-001")

    await service.handle_turn(conversation_id, "")
    await service.handle_turn(conversation_id, "Yes, I lead sales ops.")
    state = await service.handle_turn(conversation_id, "We already have a process.")

    assert state["current_node"] == "pitch"
    assert state["next_node"] == "handle_objection"

    state = await service.handle_turn(conversation_id, "Maybe later, send info.")
    assert state["current_node"] == "handle_objection"
    assert state["next_node"] in {"pitch", "wrap_up"}


@pytest.mark.asyncio
async def test_handle_turn_serializes_concurrent_updates(conversation_service):
    service, _, _, _ = conversation_service
    conversation_id = await service.start_session("lead-001")

    initial_state = await service.get_state(conversation_id)
    initial_state["current_node"] = "qualify_lead"
    initial_state["next_node"] = "qualify_lead"
    initial_state["route_decision"] = "qualify_lead"
    await service.dependencies.session_store.save(conversation_id, initial_state)

    entered = asyncio.Event()
    release = asyncio.Event()
    original_ainvoke = service.graph.ainvoke

    async def gated_ainvoke(state):
        if state["last_human_message"] == "first":
            entered.set()
            await release.wait()
        return await original_ainvoke(state)

    service.graph.ainvoke = gated_ainvoke

    first_task = asyncio.create_task(service.handle_turn(conversation_id, "first"))
    await entered.wait()
    second_task = asyncio.create_task(service.handle_turn(conversation_id, "second"))
    await asyncio.sleep(0)

    mid_state = await service.get_state(conversation_id)
    assert mid_state["turn_count"] == 0
    assert mid_state["transcript"] == []

    release.set()
    await first_task
    await second_task

    final_state = await service.get_state(conversation_id)
    human_messages = [msg["content"] for msg in final_state["transcript"] if msg["role"] == "human"]
    assert human_messages == ["first", "second"]
    assert final_state["turn_count"] == 2
