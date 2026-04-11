from __future__ import annotations

from typing import Literal, TypedDict


CallOutcome = Literal[
    "meeting_booked",
    "follow_up_needed",
    "not_interested",
    "no_answer",
    "voicemail",
]


class TranscriptMessage(TypedDict):
    role: Literal["human", "agent"]
    content: str


class SlotPayload(TypedDict):
    slot_id: str
    start_time: str
    end_time: str
    label: str


class ConversationState(TypedDict):
    lead_id: str
    lead_name: str
    lead_email: str
    phone_number: str
    company: str
    calendar_id: str
    lead_context: str
    transcript: list[TranscriptMessage]
    current_node: str
    next_node: str
    route_decision: str
    turn_count: int
    last_human_message: str
    last_agent_response: str
    is_decision_maker: bool | None
    budget_confirmed: bool | None
    timeline: str | None
    pain_points: list[str]
    available_slots: list[SlotPayload]
    proposed_slot: str | None
    meeting_booked: bool
    meeting_link: str | None
    objection_count: int
    call_outcome: CallOutcome
    follow_up_action: str | None
    qualification_notes: dict[str, str | bool | None]
    metadata: dict[str, str]


def build_initial_state(
    *,
    lead_id: str,
    lead_name: str,
    lead_email: str,
    phone_number: str,
    company: str,
    calendar_id: str,
    lead_context: str,
    available_slots: list[SlotPayload],
) -> ConversationState:
    return {
        "lead_id": lead_id,
        "lead_name": lead_name,
        "lead_email": lead_email,
        "phone_number": phone_number,
        "company": company,
        "calendar_id": calendar_id,
        "lead_context": lead_context,
        "transcript": [],
        "current_node": "start",
        "next_node": "greeting",
        "route_decision": "greeting",
        "turn_count": 0,
        "last_human_message": "",
        "last_agent_response": "",
        "is_decision_maker": None,
        "budget_confirmed": None,
        "timeline": None,
        "pain_points": [],
        "available_slots": available_slots,
        "proposed_slot": None,
        "meeting_booked": False,
        "meeting_link": None,
        "objection_count": 0,
        "call_outcome": "follow_up_needed",
        "follow_up_action": None,
        "qualification_notes": {},
        "metadata": {},
    }
