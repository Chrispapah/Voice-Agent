from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from loguru import logger

from ai_sdr_agent.graph.dynamic_nodes import (
    adjacency_for_bot_config,
    make_graph_agent_node,
    make_single_agent_node,
)
from ai_sdr_agent.graph.spec import (
    SINGLE_AGENT_NODE_ID,
    graph_execution_kind,
    parse_conversation_spec,
    require_conversation_spec,
)
from ai_sdr_agent.graph.state import ConversationState
from ai_sdr_agent.services.brain import ConversationBrain


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


def build_compiled_graph(
    *,
    brain: ConversationBrain,
    bot_config: dict[str, Any] | None,
):
    """Compile single-agent or custom graph from ``bot_config.conversation_spec``."""
    cfg = bot_config or {}
    kind = graph_execution_kind(cfg)
    if kind == "single":
        return _build_single_agent_graph(brain=brain)
    return _build_custom_graph(brain=brain, bot_config=cfg)


def _build_single_agent_graph(*, brain: ConversationBrain):
    graph = StateGraph(ConversationState)
    graph.add_node("route_turn", _route_turn)
    graph.add_node(SINGLE_AGENT_NODE_ID, make_single_agent_node(brain))
    graph.add_node("complete", lambda state: {})

    graph.add_edge(START, "route_turn")
    graph.add_conditional_edges(
        "route_turn",
        _route_decision,
        {
            SINGLE_AGENT_NODE_ID: SINGLE_AGENT_NODE_ID,
            "complete": "complete",
        },
    )
    graph.add_edge(SINGLE_AGENT_NODE_ID, END)
    graph.add_edge("complete", END)
    return graph.compile()


def _build_custom_graph(*, brain: ConversationBrain, bot_config: dict[str, Any]):
    spec = require_conversation_spec(bot_config.get("conversation_spec"))
    if spec.mode != "graph":
        raise ValueError("custom graph requires graph-mode conversation_spec")

    adj = adjacency_for_bot_config(bot_config)
    graph = StateGraph(ConversationState)
    graph.add_node("route_turn", _route_turn)
    graph.add_node("complete", lambda state: {})

    mapping: dict[str, str] = {"complete": "complete"}
    for n in spec.nodes:
        graph.add_node(n.id, make_graph_agent_node(brain, n.id, adj))
        graph.add_edge(n.id, END)
        mapping[n.id] = n.id

    graph.add_edge(START, "route_turn")
    graph.add_conditional_edges("route_turn", _route_decision, mapping)
    graph.add_edge("complete", END)
    return graph.compile()
