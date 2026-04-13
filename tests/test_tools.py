import pytest

from ai_sdr_agent.services.persistence import InMemoryLeadRepository
from ai_sdr_agent.tools import (
    StubCRMGateway,
    StubCalendarGateway,
    StubEmailGateway,
    build_calendar_tools,
    build_crm_tools,
    build_email_tools,
)


@pytest.mark.asyncio
async def test_stub_calendar_and_tools_round_trip():
    gateway = StubCalendarGateway()
    tools = {tool.name: tool for tool in build_calendar_tools(gateway)}

    slots = await tools["check_calendar_availability"].ainvoke(
        {"calendar_id": "sales-team", "date_range": "next_5_business_days"}
    )
    assert len(slots) == 3

    booking = await tools["book_calendar_event"].ainvoke(
        {
            "calendar_id": "sales-team",
            "slot_id": slots[0]["slot_id"],
            "attendee_email": "buyer@example.com",
            "title": "Demo",
            "description": "Discovery meeting",
        }
    )
    assert booking["event_id"].startswith("evt-")


@pytest.mark.asyncio
async def test_stub_email_tool_records_sent_messages():
    gateway = StubEmailGateway()
    tools = {tool.name: tool for tool in build_email_tools(gateway)}

    result = await tools["send_follow_up_email"].ainvoke(
        {
            "to_email": "buyer@example.com",
            "subject": "Next steps",
            "body": "<p>Hello</p>",
            "from_name": "AI SDR",
        }
    )
    assert result["message_id"] == "stub-email-1"
    assert len(gateway.sent_messages) == 1


@pytest.mark.asyncio
async def test_stub_crm_tool_records_outcome():
    lead_repository = InMemoryLeadRepository()
    gateway = StubCRMGateway(seed_leads=[lead_repository._leads["lead-001"]])
    tools = {tool.name: tool for tool in build_crm_tools(gateway)}

    result = await tools["update_crm"].ainvoke(
        {
            "lead_id": "lead-001",
            "call_outcome": "meeting_booked",
            "notes": "Booked for tomorrow",
        }
    )
    assert result["call_outcome"] == "meeting_booked"
    assert len(gateway.updates) == 1
