from __future__ import annotations

from ai_sdr_agent.graph.prompts import (
    BOOKING_ROUTER_PROMPT,
    OBJECTION_ROUTER_PROMPT,
    PITCH_ROUTER_PROMPT,
    QUALIFY_ROUTER_PROMPT,
)
from ai_sdr_agent.graph.state import ConversationState
from ai_sdr_agent.services.brain import ConversationBrain


def get_last_human_message(state: ConversationState) -> str:
    for message in reversed(state["transcript"]):
        if message["role"] == "human":
            return message["content"]
    return state.get("last_human_message", "")


async def route_after_qualify(
    state: ConversationState,
    brain: ConversationBrain,
) -> str:
    return await brain.classify(
        instruction=QUALIFY_ROUTER_PROMPT,
        human_input=get_last_human_message(state),
        labels=("pitch", "not_interested"),
    )


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
