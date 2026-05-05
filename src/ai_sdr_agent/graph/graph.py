from __future__ import annotations

from pathlib import Path

from langgraph.graph import END, START, StateGraph
from loguru import logger

from ai_sdr_agent.graph.nodes import (
    book_meeting_node,
    greeting_node,
    objection_node,
    pitch_node,
    qualify_node,
    wrap_up_node,
)
from ai_sdr_agent.graph.state import ConversationState
from ai_sdr_agent.services.brain import ConversationBrain
from ai_sdr_agent.tools import CRMGateway, CalendarGateway, EmailGateway


def build_sdr_graph(
    *,
    brain: ConversationBrain,
    calendar_gateway: CalendarGateway,
    email_gateway: EmailGateway,
    crm_gateway: CRMGateway,
    email_template_path: Path,
    sales_rep_name: str,
    from_name: str,
):
    graph = StateGraph(ConversationState)

    async def greeting_step(state: ConversationState) -> dict:
        return await greeting_node(state, brain=brain)

    async def qualify_step(state: ConversationState) -> dict:
        return await qualify_node(state, brain=brain)

    async def pitch_step(state: ConversationState) -> dict:
        return await pitch_node(state, brain=brain)

    async def objection_step(state: ConversationState) -> dict:
        return await objection_node(state, brain=brain)

    async def booking_step(state: ConversationState) -> dict:
        return await book_meeting_node(
            state,
            brain=brain,
            calendar_gateway=calendar_gateway,
        )

    async def wrap_up_step(state: ConversationState) -> dict:
        return await wrap_up_node(
            state,
            brain=brain,
            email_gateway=email_gateway,
            crm_gateway=crm_gateway,
            email_template_path=email_template_path,
            sales_rep_name=sales_rep_name,
            from_name=from_name,
        )

    graph.add_node("route_turn", _route_turn)
    graph.add_node("greeting", greeting_step)
    graph.add_node("qualify_lead", qualify_step)
    graph.add_node("pitch", pitch_step)
    graph.add_node("handle_objection", objection_step)
    graph.add_node("book_meeting", booking_step)
    graph.add_node("wrap_up", wrap_up_step)
    graph.add_node("complete", lambda state: {})

    graph.add_edge(START, "route_turn")
    graph.add_conditional_edges(
        "route_turn",
        _route_decision,
        {
            "greeting": "greeting",
            "qualify_lead": "qualify_lead",
            "pitch": "pitch",
            "handle_objection": "handle_objection",
            "book_meeting": "book_meeting",
            "wrap_up": "wrap_up",
            "complete": "complete",
        },
    )
    for node_name in (
        "greeting",
        "qualify_lead",
        "pitch",
        "handle_objection",
        "book_meeting",
        "wrap_up",
    ):
        graph.add_edge(node_name, END)
    graph.add_edge("complete", END)

    return graph.compile()


def _route_turn(state: ConversationState) -> dict:
    target = state["next_node"]
    metadata = state.get("metadata", {})
    logger.info(
        "Routing conversation_id={} turn_id={} target_node={} last_speaker_node={}",
        metadata.get("conversation_id", "-"),
        metadata.get("turn_id", "-"),
        target,
        state.get("current_node", "start"),
    )
    return {"route_decision": target}


def _route_decision(state: ConversationState) -> str:
    return state["route_decision"]
