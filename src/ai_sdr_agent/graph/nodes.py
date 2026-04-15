from __future__ import annotations

import asyncio
import re
import time

from loguru import logger

from ai_sdr_agent.graph.prompts import (
    booking_prompt,
    format_reply_for_tts,
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

_DEFAULT_MAX_QUALIFY_ATTEMPTS = 3
_DEFAULT_MAX_BOOKING_ATTEMPTS = 3

# Per-turn output token caps for voice (keep low for fast TTS first byte).
_MAX_OUT_GREETING = 110
_MAX_OUT_QUALIFY = 130
_MAX_OUT_PITCH = 160
_MAX_OUT_OBJECTION = 140
_MAX_OUT_BOOKING = 200
_MAX_OUT_WRAPUP = 120


def _finalize_spoken_reply(text: str) -> str:
    return format_reply_for_tts(text)


def _bot_cfg(state: ConversationState, key: str, default=None):
    return state.get("bot_config", {}).get(key, default)


def _trace_value(state: ConversationState, key: str, default: str = "-") -> str:
    value = state.get("metadata", {}).get(key)
    if value is None:
        return default
    return str(value)


def _llm_trace(state: ConversationState, node_name: str, step: str) -> dict[str, str | int]:
    return {
        "conversation_id": _trace_value(state, "conversation_id"),
        "turn_id": _trace_value(state, "turn_id"),
        "turn_count": state["turn_count"],
        "node": node_name,
        "step": step,
    }


def _log_node_start(state: ConversationState, node_name: str) -> None:
    logger.info(
        "Node start conversation_id={} turn_id={} turn_count={} node={} "
        "current_node={} transcript_messages={}",
        _trace_value(state, "conversation_id"),
        _trace_value(state, "turn_id"),
        state["turn_count"],
        node_name,
        state.get("current_node", "start"),
        len(state["transcript"]),
    )


def _log_node_fanout(state: ConversationState, node_name: str, *tasks: str) -> None:
    logger.info(
        "Node LLM fanout conversation_id={} turn_id={} turn_count={} node={} tasks={}",
        _trace_value(state, "conversation_id"),
        _trace_value(state, "turn_id"),
        state["turn_count"],
        node_name,
        ",".join(tasks),
    )


def _log_node_end(
    state: ConversationState,
    node_name: str,
    next_node: str,
    latency_ms: float,
) -> None:
    logger.info(
        "Node end conversation_id={} turn_id={} turn_count={} node={} "
        "next_node={} latency_ms={:.0f}",
        _trace_value(state, "conversation_id"),
        _trace_value(state, "turn_id"),
        state["turn_count"],
        node_name,
        next_node,
        latency_ms,
    )


def _format_details(**details: object) -> str:
    parts: list[str] = []
    for key, value in details.items():
        if value is None:
            continue
        if isinstance(value, str):
            parts.append(f"{key}={value!r}")
        else:
            parts.append(f"{key}={value}")
    return " ".join(parts)


def _log_step_latency(
    state: ConversationState,
    node_name: str,
    step_name: str,
    latency_ms: float,
    **details: object,
) -> None:
    detail_text = _format_details(**details)
    if detail_text:
        logger.info(
            "Step latency conversation_id={} turn_id={} turn_count={} node={} "
            "step={} latency_ms={:.0f} details={}",
            _trace_value(state, "conversation_id"),
            _trace_value(state, "turn_id"),
            state["turn_count"],
            node_name,
            step_name,
            latency_ms,
            detail_text,
        )
        return
    logger.info(
        "Step latency conversation_id={} turn_id={} turn_count={} node={} "
        "step={} latency_ms={:.0f}",
        _trace_value(state, "conversation_id"),
        _trace_value(state, "turn_id"),
        state["turn_count"],
        node_name,
        step_name,
        latency_ms,
    )


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
    _log_node_start(state, "greeting")
    t0 = time.perf_counter()
    response = _finalize_spoken_reply(
        await brain.respond(
            system_prompt=greeting_prompt(state),
            transcript=state["transcript"],
            max_tokens=_MAX_OUT_GREETING,
            trace=_llm_trace(state, "greeting", "respond"),
        )
    )
    respond_ms = (time.perf_counter() - t0) * 1000
    _log_step_latency(state, "greeting", "respond", respond_ms)
    logger.info("greeting_node latency respond_ms={:.0f}", respond_ms)
    _log_node_end(state, "greeting", "qualify_lead", respond_ms)
    return _append_agent_message(state, response, "greeting", "qualify_lead")


async def qualify_node(
    state: ConversationState,
    *,
    brain: ConversationBrain,
) -> dict:
    _log_node_start(state, "qualify_lead")
    new_attempts = state["qualify_attempts"] + 1

    t0 = time.perf_counter()
    _log_node_fanout(
        state,
        "qualify_lead",
        "extract_qualification",
        "respond",
        "route_after_qualify",
    )
    qual_updates, response, decision = await asyncio.gather(
        brain.extract_qualification(
            transcript=state["transcript"],
            existing_pain_points=state["pain_points"],
            trace=_llm_trace(state, "qualify_lead", "extract_qualification"),
        ),
        brain.respond(
            system_prompt=qualify_prompt(state),
            transcript=state["transcript"],
            max_tokens=_MAX_OUT_QUALIFY,
            trace=_llm_trace(state, "qualify_lead", "respond"),
        ),
        route_after_qualify(
            state,
            brain,
            trace=_llm_trace(state, "qualify_lead", "route_after_qualify"),
        ),
    )
    wall_ms = (time.perf_counter() - t0) * 1000

    _log_step_latency(
        state,
        "qualify_lead",
        "parallel_fanout",
        wall_ms,
        tasks="extract_qualification,respond,route_after_qualify",
    )
    logger.info(
        "qualify_node latency wall_ms={:.0f} (extract+respond+route parallel)",
        wall_ms,
    )
    extra: dict = {**qual_updates, "qualify_attempts": new_attempts}
    response = _finalize_spoken_reply(response)

    max_qualify = _bot_cfg(state, "max_qualify_attempts", _DEFAULT_MAX_QUALIFY_ATTEMPTS)
    if decision == "not_interested":
        extra["call_outcome"] = "not_interested"
        next_node = "wrap_up"
    elif decision == "pitch" or new_attempts >= max_qualify:
        next_node = "pitch"
    else:
        next_node = "qualify_lead"

    _log_node_end(state, "qualify_lead", next_node, wall_ms)
    return _append_agent_message(state, response, "qualify_lead", next_node, **extra)


async def pitch_node(
    state: ConversationState,
    *,
    brain: ConversationBrain,
) -> dict:
    _log_node_start(state, "pitch")
    t0 = time.perf_counter()
    _log_node_fanout(state, "pitch", "respond", "route_after_pitch")
    response, decision = await asyncio.gather(
        brain.respond(
            system_prompt=pitch_prompt(state),
            transcript=state["transcript"],
            max_tokens=_MAX_OUT_PITCH,
            trace=_llm_trace(state, "pitch", "respond"),
        ),
        route_after_pitch(
            state,
            brain,
            trace=_llm_trace(state, "pitch", "route_after_pitch"),
        ),
    )
    wall_ms = (time.perf_counter() - t0) * 1000
    response = _finalize_spoken_reply(response)

    _log_step_latency(
        state,
        "pitch",
        "parallel_fanout",
        wall_ms,
        tasks="respond,route_after_pitch",
    )
    logger.info(
        "pitch_node latency wall_ms={:.0f} (respond+route parallel)",
        wall_ms,
    )
    next_node = {
        "book_meeting": "book_meeting",
        "handle_objection": "handle_objection",
        "wrap_up": "wrap_up",
    }[decision]
    _log_node_end(state, "pitch", next_node, wall_ms)
    return _append_agent_message(state, response, "pitch", next_node)


async def objection_node(
    state: ConversationState,
    *,
    brain: ConversationBrain,
) -> dict:
    _log_node_start(state, "handle_objection")
    new_count = state["objection_count"] + 1
    extra: dict = {"objection_count": new_count}
    max_objections = _bot_cfg(state, "max_objection_attempts", 2)

    t0 = time.perf_counter()
    if new_count >= max_objections:
        response = _finalize_spoken_reply(
            await brain.respond(
                system_prompt=objection_prompt(state),
                transcript=state["transcript"],
                max_tokens=_MAX_OUT_OBJECTION,
                trace=_llm_trace(state, "handle_objection", "respond"),
            )
        )
        wall_ms = (time.perf_counter() - t0) * 1000
        _log_step_latency(
            state,
            "handle_objection",
            "respond_only",
            wall_ms,
            reason="max_objections_reached",
        )
        logger.info("objection_node latency wall_ms={:.0f} (max objections reached)", wall_ms)
        extra["call_outcome"] = "follow_up_needed"
        next_node = "wrap_up"
    else:
        _log_node_fanout(state, "handle_objection", "respond", "route_after_objection")
        response, decision = await asyncio.gather(
            brain.respond(
                system_prompt=objection_prompt(state),
                transcript=state["transcript"],
                max_tokens=_MAX_OUT_OBJECTION,
                trace=_llm_trace(state, "handle_objection", "respond"),
            ),
            route_after_objection(
                state,
                brain,
                trace=_llm_trace(state, "handle_objection", "route_after_objection"),
            ),
        )
        wall_ms = (time.perf_counter() - t0) * 1000
        response = _finalize_spoken_reply(response)
        _log_step_latency(
            state,
            "handle_objection",
            "parallel_fanout",
            wall_ms,
            tasks="respond,route_after_objection",
        )
        logger.info(
            "objection_node latency wall_ms={:.0f} (respond+route parallel)",
            wall_ms,
        )
        next_node = "pitch" if decision == "pitch" else "wrap_up"
    _log_node_end(state, "handle_objection", next_node, wall_ms)
    return _append_agent_message(state, response, "handle_objection", next_node, **extra)


async def book_meeting_node(
    state: ConversationState,
    *,
    brain: ConversationBrain,
    calendar_gateway: CalendarGateway,
) -> dict:
    _log_node_start(state, "book_meeting")
    new_attempts = state["booking_attempts"] + 1
    extra: dict = {"booking_attempts": new_attempts}

    available_slots = state["available_slots"]
    if not available_slots:
        list_slots_t0 = time.perf_counter()
        slots = await calendar_gateway.list_available_slots(
            calendar_id=state["calendar_id"],
            date_range="next_5_business_days",
        )
        list_slots_ms = (time.perf_counter() - list_slots_t0) * 1000
        _log_step_latency(
            state,
            "book_meeting",
            "list_available_slots",
            list_slots_ms,
            slot_count=len(slots),
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

    detect_slot_t0 = time.perf_counter()
    confirmed = _detect_confirmed_slot(state)
    detect_slot_ms = (time.perf_counter() - detect_slot_t0) * 1000
    _log_step_latency(
        state,
        "book_meeting",
        "detect_confirmed_slot",
        detect_slot_ms,
        matched=confirmed is not None,
        candidate_slot_count=len(available_slots),
        state_slot_count=len(state["available_slots"]),
    )
    t0 = time.perf_counter()

    if confirmed is not None:
        book_slot_t0 = time.perf_counter()
        booking = await calendar_gateway.book_slot(
            calendar_id=state["calendar_id"],
            slot_id=confirmed["slot_id"],
            attendee_email=state["lead_email"],
            title=f"Meeting with {state['lead_name']} - {state['company']}",
            description="Follow-up discovery call booked by the AI SDR.",
        )
        book_slot_ms = (time.perf_counter() - book_slot_t0) * 1000
        _log_step_latency(
            state,
            "book_meeting",
            "book_slot",
            book_slot_ms,
            slot_id=confirmed["slot_id"],
        )
        extra.update(
            meeting_booked=True,
            proposed_slot=confirmed["start_time"],
            meeting_link=booking["link"],
            call_outcome="meeting_booked",
            follow_up_action="send_meeting_confirmation",
        )
        wall_ms = (time.perf_counter() - t0) * 1000
        _log_step_latency(
            state,
            "book_meeting",
            "confirm_booking",
            wall_ms,
            slot_id=confirmed["slot_id"],
        )
        logger.info(
            "book_meeting_node latency wall_ms={:.0f} (slot confirmed)",
            wall_ms,
        )
        response = (
            f"Great, I have you booked for {confirmed['label']}. "
            "I will send the invite and follow-up details right after this call."
        )
        next_node = "wrap_up"
    elif new_attempts >= _bot_cfg(state, "max_booking_attempts", _DEFAULT_MAX_BOOKING_ATTEMPTS):
        response = _finalize_spoken_reply(
            await brain.respond(
                system_prompt=booking_prompt(state),
                transcript=state["transcript"],
                max_tokens=_MAX_OUT_BOOKING,
                trace=_llm_trace(state, "book_meeting", "respond"),
            )
        )
        wall_ms = (time.perf_counter() - t0) * 1000
        _log_step_latency(
            state,
            "book_meeting",
            "respond_only",
            wall_ms,
            reason="attempts_exhausted",
        )
        logger.info(
            "book_meeting_node latency wall_ms={:.0f} (attempts exhausted)",
            wall_ms,
        )
        extra.update(call_outcome="follow_up_needed", follow_up_action="manual_booking_follow_up")
        next_node = "wrap_up"
    else:
        _log_node_fanout(state, "book_meeting", "respond", "route_during_booking")
        response, decision = await asyncio.gather(
            brain.respond(
                system_prompt=booking_prompt(state),
                transcript=state["transcript"],
                max_tokens=_MAX_OUT_BOOKING,
                trace=_llm_trace(state, "book_meeting", "respond"),
            ),
            route_during_booking(
                state,
                brain,
                trace=_llm_trace(state, "book_meeting", "route_during_booking"),
            ),
        )
        wall_ms = (time.perf_counter() - t0) * 1000
        response = _finalize_spoken_reply(response)
        _log_step_latency(
            state,
            "book_meeting",
            "parallel_fanout",
            wall_ms,
            tasks="respond,route_during_booking",
        )
        logger.info(
            "book_meeting_node latency wall_ms={:.0f} (respond+route parallel)",
            wall_ms,
        )
        if decision == "wrap_up":
            extra.update(call_outcome="follow_up_needed", follow_up_action="manual_booking_follow_up")
            next_node = "wrap_up"
        else:
            next_node = "book_meeting"
    _log_node_end(state, "book_meeting", next_node, wall_ms)
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
    _log_node_start(state, "wrap_up")
    t0 = time.perf_counter()
    response = _finalize_spoken_reply(
        await brain.respond(
            system_prompt=wrap_up_prompt(state),
            transcript=state["transcript"],
            max_tokens=_MAX_OUT_WRAPUP,
            trace=_llm_trace(state, "wrap_up", "respond"),
        )
    )
    respond_ms = (time.perf_counter() - t0) * 1000
    _log_step_latency(state, "wrap_up", "respond", respond_ms)
    summary = (
        f"Outcome: {state['call_outcome']}. "
        f"Meeting booked: {'yes' if state['meeting_booked'] else 'no'}."
    )
    if state["meeting_booked"] or state["follow_up_action"]:
        render_email_t0 = time.perf_counter()
        html = render_follow_up_email(
            template_path=email_template_path,
            lead_name=state["lead_name"],
            company=state["company"],
            sales_rep_name=sales_rep_name,
            meeting_slot=state["proposed_slot"],
            follow_up_summary=summary,
        )
        render_email_ms = (time.perf_counter() - render_email_t0) * 1000
        _log_step_latency(state, "wrap_up", "render_follow_up_email", render_email_ms)
        send_email_t0 = time.perf_counter()
        await email_gateway.send_email(
            to_email=state["lead_email"],
            subject=f"Follow-up from {sales_rep_name}",
            body=html,
            from_name=from_name,
        )
        send_email_ms = (time.perf_counter() - send_email_t0) * 1000
        _log_step_latency(
            state,
            "wrap_up",
            "send_email",
            send_email_ms,
            follow_up_action=state["follow_up_action"],
        )
    update_crm_t0 = time.perf_counter()
    await crm_gateway.update_call_outcome(
        lead_id=state["lead_id"],
        call_outcome=state["call_outcome"],
        notes=summary,
    )
    update_crm_ms = (time.perf_counter() - update_crm_t0) * 1000
    _log_step_latency(
        state,
        "wrap_up",
        "update_crm",
        update_crm_ms,
        call_outcome=state["call_outcome"],
    )
    side_effects_ms = (time.perf_counter() - t0) * 1000 - respond_ms
    _log_step_latency(state, "wrap_up", "side_effects_total", side_effects_ms)
    logger.info(
        "wrap_up_node latency respond_ms={:.0f} side_effects_ms={:.0f} total_ms={:.0f}",
        respond_ms, side_effects_ms, respond_ms + side_effects_ms,
    )
    _log_node_end(state, "wrap_up", "complete", respond_ms + side_effects_ms)
    return _append_agent_message(state, response, "wrap_up", "complete")
