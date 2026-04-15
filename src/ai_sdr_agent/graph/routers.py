from __future__ import annotations

import re

from loguru import logger

from ai_sdr_agent.graph.prompts import (
    BOOKING_ROUTER_PROMPT,
    OBJECTION_ROUTER_PROMPT,
    PITCH_ROUTER_PROMPT,
    QUALIFY_ROUTER_PROMPT,
)
from ai_sdr_agent.graph.state import ConversationState
from ai_sdr_agent.services.brain import ConversationBrain

# If the LLM labels a turn as not_interested, only keep that when the transcript
# shows an explicit refusal. Otherwise noisy phone STT (e.g. "yeah yeah don't"
# clipped from "go ahead") can wrongly end qualification while the agent still
# sounds engaged.
_STRONG_DISINTEREST = re.compile(
    r"\b("
    r"not interested|no thanks|no thank you|no thankyou|"
    r"stop calling|don'?t call|do not call|remove me|take me off|"
    r"not for us|wrong (person|number|time)|"
    r"don'?t want|we'?ll pass|i'?ll pass|hard pass|"
    r"go away|leave me alone|hang up|goodbye|bye bye"
    r")\b",
    re.IGNORECASE,
)

_SOFT_ENGAGEMENT = re.compile(
    r"\b("
    r"y(es|eah|ep)|sure|ok(ay)?|go ahead|do it|tell me|please|of course|"
    r"that'?s fine|sounds good|let'?s hear"
    r")\b",
    re.IGNORECASE,
)

_BOOKING_SIGNAL = re.compile(
    r"\b("
    r"yes|yeah|yep|sure|ok(ay)?|works|that works|book it|let'?s do it|"
    r"schedule it|set it up|tomorrow|monday|tuesday|wednesday|thursday|friday|"
    r"am|pm|afternoon|morning|first|second|third|option"
    r")\b",
    re.IGNORECASE,
)

_PITCH_BOOKING_SIGNAL = re.compile(
    r"\b("
    r"yes|yeah|yep|sure|ok(ay)?|sounds good|let'?s do it|book|schedule|"
    r"demo|walkthrough|show me|tell me more"
    r")\b",
    re.IGNORECASE,
)

_OBJECTION_SIGNAL = re.compile(
    r"\b("
    r"already have|already using|busy|send info|maybe later|not right now|"
    r"just send|email me|we have a process|we use"
    r")\b",
    re.IGNORECASE,
)

def _trace_value(trace: dict[str, object] | None, key: str, default: str = "-") -> str:
    if not trace:
        return default
    value = trace.get(key)
    if value is None:
        return default
    return str(value)


def apply_qualify_router_safety(human_input: str, decision: str) -> str:
    if decision != "not_interested":
        return decision
    text = (human_input or "").strip()
    if not text:
        return decision
    if _STRONG_DISINTEREST.search(text):
        return "not_interested"
    if _SOFT_ENGAGEMENT.search(text):
        logger.info(
            "Overriding qualify router not_interested -> continue_qualifying (soft engagement, no explicit refusal) text={!r}",
            text,
        )
        return "continue_qualifying"
    return decision


def get_last_human_message(state: ConversationState) -> str:
    for message in reversed(state["transcript"]):
        if message["role"] == "human":
            return message["content"]
    return state.get("last_human_message", "")


def _qualify_fast_path(state: ConversationState, human: str) -> str | None:
    text = (human or "").strip()
    if not text:
        return None
    if _STRONG_DISINTEREST.search(text):
        return "not_interested"
    return None


def _pitch_fast_path(human: str) -> str | None:
    text = (human or "").strip()
    if not text:
        return None
    if _STRONG_DISINTEREST.search(text):
        return "wrap_up"
    if _PITCH_BOOKING_SIGNAL.search(text):
        return "book_meeting"
    if _OBJECTION_SIGNAL.search(text):
        return "handle_objection"
    return None


def _objection_fast_path(human: str) -> str | None:
    text = (human or "").strip()
    if not text:
        return None
    if _STRONG_DISINTEREST.search(text):
        return "wrap_up"
    if _SOFT_ENGAGEMENT.search(text):
        return "pitch"
    return None


def _booking_fast_path(human: str) -> str | None:
    text = (human or "").strip()
    if not text:
        return None
    if _STRONG_DISINTEREST.search(text):
        return "wrap_up"
    if _BOOKING_SIGNAL.search(text):
        return "continue_booking"
    return None


async def route_after_qualify(
    state: ConversationState,
    brain: ConversationBrain,
    *,
    trace: dict[str, object] | None = None,
) -> str:
    human = get_last_human_message(state)
    fast_path = _qualify_fast_path(state, human)
    if fast_path is not None:
        logger.info(
            "Router decision conversation_id={} turn_id={} turn_count={} node={} "
            "router=qualify decision={} source=fast_path",
            _trace_value(trace, "conversation_id"),
            _trace_value(trace, "turn_id"),
            _trace_value(trace, "turn_count"),
            _trace_value(trace, "node"),
            fast_path,
        )
        return fast_path
    decision = await brain.classify(
        instruction=QUALIFY_ROUTER_PROMPT,
        human_input=human,
        labels=("continue_qualifying", "pitch", "not_interested"),
        trace=trace,
    )
    safe_decision = apply_qualify_router_safety(human, decision)
    logger.info(
        "Router decision conversation_id={} turn_id={} turn_count={} node={} "
        "router=qualify raw_decision={} final_decision={}",
        _trace_value(trace, "conversation_id"),
        _trace_value(trace, "turn_id"),
        _trace_value(trace, "turn_count"),
        _trace_value(trace, "node"),
        decision,
        safe_decision,
    )
    return safe_decision


async def route_after_pitch(
    state: ConversationState,
    brain: ConversationBrain,
    *,
    trace: dict[str, object] | None = None,
) -> str:
    human = get_last_human_message(state)
    fast_path = _pitch_fast_path(human)
    if fast_path is not None:
        logger.info(
            "Router decision conversation_id={} turn_id={} turn_count={} node={} "
            "router=pitch decision={} source=fast_path",
            _trace_value(trace, "conversation_id"),
            _trace_value(trace, "turn_id"),
            _trace_value(trace, "turn_count"),
            _trace_value(trace, "node"),
            fast_path,
        )
        return fast_path
    decision = await brain.classify(
        instruction=PITCH_ROUTER_PROMPT,
        human_input=human,
        labels=("book_meeting", "handle_objection", "wrap_up"),
        trace=trace,
    )
    logger.info(
        "Router decision conversation_id={} turn_id={} turn_count={} node={} "
        "router=pitch decision={}",
        _trace_value(trace, "conversation_id"),
        _trace_value(trace, "turn_id"),
        _trace_value(trace, "turn_count"),
        _trace_value(trace, "node"),
        decision,
    )
    return decision


async def route_after_objection(
    state: ConversationState,
    brain: ConversationBrain,
    *,
    trace: dict[str, object] | None = None,
) -> str:
    human = get_last_human_message(state)
    fast_path = _objection_fast_path(human)
    if fast_path is not None:
        logger.info(
            "Router decision conversation_id={} turn_id={} turn_count={} node={} "
            "router=objection decision={} source=fast_path",
            _trace_value(trace, "conversation_id"),
            _trace_value(trace, "turn_id"),
            _trace_value(trace, "turn_count"),
            _trace_value(trace, "node"),
            fast_path,
        )
        return fast_path
    decision = await brain.classify(
        instruction=OBJECTION_ROUTER_PROMPT,
        human_input=human,
        labels=("pitch", "wrap_up"),
        trace=trace,
    )
    logger.info(
        "Router decision conversation_id={} turn_id={} turn_count={} node={} "
        "router=objection decision={}",
        _trace_value(trace, "conversation_id"),
        _trace_value(trace, "turn_id"),
        _trace_value(trace, "turn_count"),
        _trace_value(trace, "node"),
        decision,
    )
    return decision


async def route_during_booking(
    state: ConversationState,
    brain: ConversationBrain,
    *,
    trace: dict[str, object] | None = None,
) -> str:
    human = get_last_human_message(state)
    fast_path = _booking_fast_path(human)
    if fast_path is not None:
        decision = "wrap_up" if fast_path == "wrap_up" else "continue_booking"
        logger.info(
            "Router decision conversation_id={} turn_id={} turn_count={} node={} "
            "router=booking final_decision={} source=fast_path",
            _trace_value(trace, "conversation_id"),
            _trace_value(trace, "turn_id"),
            _trace_value(trace, "turn_count"),
            _trace_value(trace, "node"),
            decision,
        )
        return decision
    result = await brain.classify(
        instruction=BOOKING_ROUTER_PROMPT,
        human_input=human,
        labels=("continue_booking", "wrap_up"),
        trace=trace,
    )
    decision = "wrap_up" if result == "wrap_up" else "continue_booking"
    logger.info(
        "Router decision conversation_id={} turn_id={} turn_count={} node={} "
        "router=booking raw_decision={} final_decision={}",
        _trace_value(trace, "conversation_id"),
        _trace_value(trace, "turn_id"),
        _trace_value(trace, "turn_count"),
        _trace_value(trace, "node"),
        result,
        decision,
    )
    return decision
