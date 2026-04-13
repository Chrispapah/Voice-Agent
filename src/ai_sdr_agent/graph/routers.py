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


async def route_after_qualify(
    state: ConversationState,
    brain: ConversationBrain,
) -> str:
    human = get_last_human_message(state)
    decision = await brain.classify(
        instruction=QUALIFY_ROUTER_PROMPT,
        human_input=human,
        labels=("continue_qualifying", "pitch", "not_interested"),
    )
    return apply_qualify_router_safety(human, decision)


async def route_after_pitch(
    state: ConversationState,
    brain: ConversationBrain,
) -> str:
    return await brain.classify(
        instruction=PITCH_ROUTER_PROMPT,
        human_input=get_last_human_message(state),
        labels=("book_meeting", "handle_objection", "wrap_up"),
    )


async def route_after_objection(
    state: ConversationState,
    brain: ConversationBrain,
) -> str:
    return await brain.classify(
        instruction=OBJECTION_ROUTER_PROMPT,
        human_input=get_last_human_message(state),
        labels=("pitch", "wrap_up"),
    )


async def route_during_booking(
    state: ConversationState,
    brain: ConversationBrain,
) -> str:
    result = await brain.classify(
        instruction=BOOKING_ROUTER_PROMPT,
        human_input=get_last_human_message(state),
        labels=("continue_booking", "wrap_up"),
    )
    return "wrap_up" if result == "wrap_up" else "continue_booking"
