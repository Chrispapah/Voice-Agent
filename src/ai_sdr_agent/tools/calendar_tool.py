from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Protocol

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class MeetingSlot(BaseModel):
    slot_id: str
    start_time: datetime
    end_time: datetime
    label: str


class CalendarAvailabilityRequest(BaseModel):
    calendar_id: str
    date_range: str = "next_5_business_days"


class CalendarBookingRequest(BaseModel):
    calendar_id: str
    slot_id: str
    attendee_email: str
    title: str
    description: str


class CalendarGateway(Protocol):
    async def list_available_slots(
        self,
        *,
        calendar_id: str,
        date_range: str,
    ) -> list[MeetingSlot]:
        ...

    async def book_slot(
        self,
        *,
        calendar_id: str,
        slot_id: str,
        attendee_email: str,
        title: str,
        description: str,
    ) -> dict[str, str]:
        ...


class StubCalendarGateway:
    def __init__(self):
        today = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        self._slots = [
            MeetingSlot(
                slot_id="slot-1",
                start_time=today + timedelta(days=1, hours=15),
                end_time=today + timedelta(days=1, hours=15, minutes=30),
                label="Tomorrow at 3:00 PM UTC",
            ),
            MeetingSlot(
                slot_id="slot-2",
                start_time=today + timedelta(days=2, hours=10),
                end_time=today + timedelta(days=2, hours=10, minutes=30),
                label="In two days at 10:00 AM UTC",
            ),
            MeetingSlot(
                slot_id="slot-3",
                start_time=today + timedelta(days=3, hours=17),
                end_time=today + timedelta(days=3, hours=17, minutes=30),
                label="In three days at 5:00 PM UTC",
            ),
        ]
        self.bookings: list[dict[str, str]] = []

    async def list_available_slots(
        self,
        *,
        calendar_id: str,
        date_range: str,
    ) -> list[MeetingSlot]:
        return self._slots

    async def book_slot(
        self,
        *,
        calendar_id: str,
        slot_id: str,
        attendee_email: str,
        title: str,
        description: str,
    ) -> dict[str, str]:
        booking = {
            "calendar_id": calendar_id,
            "slot_id": slot_id,
            "attendee_email": attendee_email,
            "title": title,
            "description": description,
            "event_id": f"evt-{slot_id}",
            "link": f"https://calendar.example.com/{slot_id}",
        }
        self.bookings.append(booking)
        return booking


def build_calendar_tools(calendar_gateway: CalendarGateway) -> list[StructuredTool]:
    async def check_calendar_availability(
        calendar_id: str,
        date_range: str = "next_5_business_days",
    ) -> list[dict[str, str]]:
        slots = await calendar_gateway.list_available_slots(
            calendar_id=calendar_id,
            date_range=date_range,
        )
        return [
            {
                "slot_id": slot.slot_id,
                "start_time": slot.start_time.isoformat(),
                "end_time": slot.end_time.isoformat(),
                "label": slot.label,
            }
            for slot in slots
        ]

    async def book_calendar_event(
        calendar_id: str,
        slot_id: str,
        attendee_email: str,
        title: str,
        description: str,
    ) -> dict[str, str]:
        return await calendar_gateway.book_slot(
            calendar_id=calendar_id,
            slot_id=slot_id,
            attendee_email=attendee_email,
            title=title,
            description=description,
        )

    return [
        StructuredTool.from_function(
            coroutine=check_calendar_availability,
            name="check_calendar_availability",
            description="Check available meeting slots for an SDR follow-up meeting.",
            args_schema=CalendarAvailabilityRequest,
        ),
        StructuredTool.from_function(
            coroutine=book_calendar_event,
            name="book_calendar_event",
            description="Book a meeting slot on the calendar for the prospect.",
            args_schema=CalendarBookingRequest,
        ),
    ]
