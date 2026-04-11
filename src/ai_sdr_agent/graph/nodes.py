from __future__ import annotations

import re

from loguru import logger

from ai_sdr_agent.graph.prompts import (
    booking_prompt,
    greeting_prompt,
    objection_prompt,
    pitch_prompt,
    qualify_prompt,
    wrap_up_prompt,
)
from ai_sdr_agent.graph.routers import (
    get_last_human_message,
    route_after_objection,
    route_after_pitch,
    route_after_qualify,
    route_during_booking,
)
from ai_sdr_agent.graph.state import ConversationState
from ai_sdr_agent.services.brain import ConversationBrain
from ai_sdr_agent.tools import CRMGateway, CalendarGateway, EmailGateway, render_follow_up_email

MAX_QUALIFY_ATTEMPTS = 3
MAX_BOOKING_ATTEMPTS = 3


def _append_agent_message(
    state: ConversationState,
    content: str,
    node_name: str,
    next_node: str,
    **extra_fields,
) -> dict:
    update: dict = {
        "transcript": state["transcript"] + [{"role": "agent", "content": content}],
        "current_node": node_name,
        "last_agent_response": content,
        "next_node": next_node,
        "route_decision": next_node,
    }
    update.update(extra_fields)
    return update


_ORDINAL_MAP: list[tuple[re.Pattern[str], int]] = [
    (re.compile(r"\b(first|1st|number\s*one|option\s*one|option\s*1|the\s*one)\b", re.I), 0),
    (re.compile(r"\b(second|2nd|number\s*two|option\s*two|option\s*2)\b", re.I), 1),
    (re.compile(r"\b(third|3rd|number\s*three|option\s*three|option\s*3|last)\b", re.I), 2),
]

_TIME_KEYWORDS: list[tuple[re.Pattern[str], int]] = [
    (re.compile(r"\btomorrow\b", re.I), 0),
    (re.compile(r"\b(two|2)\s*day", re.I), 1),
    (re.compile(r"\b(three|3)\s*day", re.I), 2),
    (re.compile(r"\b3\s*(:00\s*)?p\.?m\b", re.I), 0),
    (re.compile(r"\b10\s*(:00\s*)?a\.?m\b", re.I), 1),
    (re.compile(r"\b5\s*(:00\s*)?p\.?m\b", re.I), 2),
]


def _detect_confirmed_slot(state: ConversationState) -> dict[str, str] | None:
    slots = state["available_slots"]
    if not slots:
        return None
    last_human = get_last_human_message(state).lower().strip()
    if not last_human:
        return None

    for slot in slots:
        label = slot["label"].lower()
        start_time = slot["start_time"].lower()
        if slot["slot_id"].lower() in last_human:
            return slot
        if label in last_human:
            return slot
        simple_label = label.replace("utc", "").replace("in ", "").strip()
        if simple_label and simple_label in last_human:
            return slot
        if start_time[:16] in last_human:
            return slot

    for pattern, idx in _ORDINAL_MAP:
        if pattern.search(last_human) and idx < len(slots):
            logger.info("Slot matched via ordinal index={} text={!r}", idx, last_human)
            return slots[idx]

    for pattern, idx in _TIME_KEYWORDS:
        if pattern.search(last_human) and idx < len(slots):
            logger.info("Slot matched via time keyword index={} text={!r}", idx, last_human)
            return slots[idx]

    return None


async def greeting_node(
    state: ConversationState,
    *,
    brain: ConversationBrain,
) -> dict:
    response = await brain.respond(
        system_prompt=greeting_prompt(state),
        transcript=state["transcript"],
    )
    return _append_agent_message(state, response, "greeting", "qualify_lead")


async def qualify_node(
    state: ConversationState,
    *,
    brain: ConversationBrain,
) -> dict:
    new_attempts = state["qualify_attempts"] + 1
    qual_updates = await brain.extract_qualification(
        transcript=state["transcript"],
        existing_pain_points=state["pain_points"],
    )
    response = await brain.respond(
        system_prompt=qualify_prompt(state),
        transcript=state["transcript"],
    )
    decision = await route_after_qualify(state, brain)
    extra: dict = {**qual_updates, "qualify_attempts": new_attempts}

    if decision == "not_interested":
        extra["call_outcome"] = "not_interested"
        next_node = "wrap_up"
    elif decision == "pitch" or new_attempts >= MAX_QUALIFY_ATTEMPTS:
        next_node = "pitch"
    else:
        next_node = "qualify_lead"

    return _append_agent_message(state, response, "qualify_lead", next_node, **extra)


async def pitch_node(
    state: ConversationState,
    *,
    brain: ConversationBrain,
) -> dict:
    response = await brain.respond(
        system_prompt=pitch_prompt(state),
        transcript=state["transcript"],
    )
    decision = await route_after_pitch(state, brain)
    next_node = {
        "book_meeting": "book_meeting",
        "handle_objection": "handle_objection",
        "wrap_up": "wrap_up",
    }[decision]
    return _append_agent_message(state, response, "pitch", next_node)


async def objection_node(
    state: ConversationState,
    *,
    brain: ConversationBrain,
) -> dict:
    new_count = state["objection_count"] + 1
    response = await brain.respond(
        system_prompt=objection_prompt(state),
        transcript=state["transcript"],
    )
    extra: dict = {"objection_count": new_count}
    if new_count >= 2:
        extra["call_outcome"] = "follow_up_needed"
        next_node = "wrap_up"
    else:
        decision = await route_after_objection(state, brain)
        next_node = "pitch" if decision == "pitch" else "wrap_up"
    return _append_agent_message(state, response, "handle_objection", next_node, **extra)


async def book_meeting_node(
    state: ConversationState,
    *,
    brain: ConversationBrain,
    calendar_gateway: CalendarGateway,
) -> dict:
    new_attempts = state["booking_attempts"] + 1
    extra: dict = {"booking_attempts": new_attempts}

    available_slots = state["available_slots"]
    if not available_slots:
        slots = await calendar_gateway.list_available_slots(
            calendar_id=state["calendar_id"],
            date_range="next_5_business_days",
        )
        available_slots = [
            {
                "slot_id": slot.slot_id,
                "start_time": slot.start_time.isoformat(),
                "end_time": slot.end_time.isoformat(),
                "label": slot.label,
            }
            for slot in slots
        ]
        extra["available_slots"] = available_slots

    response = await brain.respond(
        system_prompt=booking_prompt(state),
        transcript=state["transcript"],
    )
    confirmed = _detect_confirmed_slot(state)
    if confirmed is not None:
        booking = await calendar_gateway.book_slot(
            calendar_id=state["calendar_id"],
            slot_id=confirmed["slot_id"],
            attendee_email=state["lead_email"],
            title=f"Meeting with {state['lead_name']} - {state['company']}",
            description="Follow-up discovery call booked by the AI SDR.",
        )
        extra.update(
            meeting_booked=True,
            proposed_slot=confirmed["start_time"],
            meeting_link=booking["link"],
            call_outcome="meeting_booked",
            follow_up_action="send_meeting_confirmation",
        )
        response = (
            f"Great, I have you booked for {confirmed['label']}. "
            "I will send the invite and follow-up details right after this call."
        )
        next_node = "wrap_up"
    elif new_attempts >= MAX_BOOKING_ATTEMPTS:
        logger.info("Booking attempts exhausted, wrapping up")
        extra.update(call_outcome="follow_up_needed", follow_up_action="manual_booking_follow_up")
        next_node = "wrap_up"
    else:
        decision = await route_during_booking(state, brain)
        if decision == "wrap_up":
            extra.update(call_outcome="follow_up_needed", follow_up_action="manual_booking_follow_up")
            next_node = "wrap_up"
        else:
            next_node = "book_meeting"
    return _append_agent_message(state, response, "book_meeting", next_node, **extra)


async def wrap_up_node(
    state: ConversationState,
    *,
    brain: ConversationBrain,
    email_gateway: EmailGateway,
    crm_gateway: CRMGateway,
    email_template_path,
    sales_rep_name: str,
    from_name: str,
) -> dict:
    response = await brain.respond(
        system_prompt=wrap_up_prompt(state),
        transcript=state["transcript"],
    )
    summary = (
        f"Outcome: {state['call_outcome']}. "
        f"Meeting booked: {'yes' if state['meeting_booked'] else 'no'}."
    )
    if state["meeting_booked"] or state["follow_up_action"]:
        html = render_follow_up_email(
            template_path=email_template_path,
            lead_name=state["lead_name"],
            company=state["company"],
            sales_rep_name=sales_rep_name,
            meeting_slot=state["proposed_slot"],
            follow_up_summary=summary,
        )
        await email_gateway.send_email(
            to_email=state["lead_email"],
            subject=f"Follow-up from {sales_rep_name}",
            body=html,
            from_name=from_name,
        )
    await crm_gateway.update_call_outcome(
        lead_id=state["lead_id"],
        call_outcome=state["call_outcome"],
        notes=summary,
    )
    return _append_agent_message(state, response, "wrap_up", "complete")
