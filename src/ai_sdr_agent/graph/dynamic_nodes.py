from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from loguru import logger

from ai_sdr_agent.graph.prompts import _VOICE_OUTPUT_RULES, _template_vars, format_reply_for_tts
from ai_sdr_agent.graph.spec import (
    SINGLE_AGENT_NODE_ID,
    build_adjacency,
    parse_conversation_spec,
    prompt_for_node,
)
from ai_sdr_agent.graph.state import ConversationState

if TYPE_CHECKING:
    from ai_sdr_agent.services.brain import ConversationBrain


def _interpolate_placeholders(template: str, state: ConversationState) -> str:
    ctx = dict(_template_vars(state))
    try:
        return template.format(**ctx)
    except (KeyError, ValueError):
        return template


def _trace(state: ConversationState) -> dict[str, Any]:
    meta = state.get("metadata", {})
    return {
        "conversation_id": meta.get("conversation_id", "-"),
        "turn_id": meta.get("turn_id", "-"),
        "turn_count": state["turn_count"],
        "node": state.get("current_node", "-"),
        "step": "dynamic",
    }


async def _pick_next_node(
    *,
    brain: ConversationBrain,
    state: ConversationState,
    current_id: str,
    outgoing: list[str],
    trace: dict[str, Any],
) -> str:
    if not outgoing:
        return current_id
    if len(outgoing) == 1:
        return outgoing[0]
    human = state.get("last_human_message", "") or ""
    instruction = (
        "You route a voice conversation between specialized agents. "
        f"The active agent was {current_id!r}. Based on the user's latest message, "
        "choose which agent should speak next. Only pick from the allowed labels."
    )
    choice = await brain.classify(
        instruction=instruction,
        human_input=human,
        labels=outgoing,
        trace=trace,
    )
    if choice in outgoing:
        return choice
    return outgoing[0]


def _append_agent(
    state: ConversationState,
    content: str,
    node_name: str,
    next_node: str,
) -> dict[str, Any]:
    return {
        "transcript": state["transcript"] + [{"role": "agent", "content": content}],
        "current_node": node_name,
        "last_agent_response": content,
        "next_node": next_node,
        "route_decision": next_node,
    }


def make_single_agent_node(
    brain: ConversationBrain,
) -> Callable[[ConversationState], Awaitable[dict[str, Any]]]:
    async def single_agent_step(state: ConversationState) -> dict[str, Any]:
        t0 = time.perf_counter()
        spec = parse_conversation_spec(state["bot_config"].get("conversation_spec"))
        if spec is None or spec.mode != "single":
            raise RuntimeError("single_agent_step requires single-mode spec")
        trace = _trace(state)
        has_human = any(m.get("role") == "human" for m in state["transcript"])
        if not has_human:
            raw = state.get("bot_config", {}).get("initial_greeting") or "Hello."
            text = format_reply_for_tts(str(raw))
            next_n = SINGLE_AGENT_NODE_ID
            logger.info("single_agent_node used initial_greeting latency_ms={:.0f}", (time.perf_counter() - t0) * 1000)
            return _append_agent(state, text, SINGLE_AGENT_NODE_ID, next_n)

        system = _interpolate_placeholders(spec.system_prompt or "", state)
        system = f"{system.strip()}\n{_VOICE_OUTPUT_RULES}"
        max_out = min(int(state.get("bot_config", {}).get("llm_max_tokens", 220) or 220), 400)
        response = format_reply_for_tts(
            await brain.respond(
                system_prompt=system,
                transcript=state["transcript"],
                max_tokens=max_out,
                trace=trace,
            )
        )
        logger.info(
            "single_agent_node LLM latency_ms={:.0f}",
            (time.perf_counter() - t0) * 1000,
        )
        return _append_agent(state, response, SINGLE_AGENT_NODE_ID, SINGLE_AGENT_NODE_ID)

    return single_agent_step


def make_graph_agent_node(
    brain: ConversationBrain,
    node_id: str,
    adjacency: dict[str, list[str]],
) -> Callable[[ConversationState], Awaitable[dict[str, Any]]]:
    async def graph_agent_step(state: ConversationState) -> dict[str, Any]:
        t0 = time.perf_counter()
        spec = parse_conversation_spec(state["bot_config"].get("conversation_spec"))
        if spec is None or spec.mode != "graph":
            raise RuntimeError("graph_agent_step requires graph-mode spec")
        trace = {**_trace(state), "node": node_id}
        has_human = any(m.get("role") == "human" for m in state["transcript"])
        outgoing = list(adjacency.get(node_id, []))
        if not has_human:
            raw = state.get("bot_config", {}).get("initial_greeting") or "Hello."
            response = format_reply_for_tts(str(raw))
            if len(outgoing) == 1:
                next_node = outgoing[0]
            elif not outgoing:
                next_node = node_id
            else:
                next_node = await _pick_next_node(
                    brain=brain,
                    state=state,
                    current_id=node_id,
                    outgoing=outgoing,
                    trace=trace,
                )
            logger.info(
                "graph_agent_node opener node={} next={} latency_ms={:.0f}",
                node_id,
                next_node,
                (time.perf_counter() - t0) * 1000,
            )
            return _append_agent(state, response, node_id, next_node)

        base_prompt = _interpolate_placeholders(prompt_for_node(spec, node_id), state)
        system = f"{base_prompt.strip()}\n{_VOICE_OUTPUT_RULES}"
        max_out = min(int(state.get("bot_config", {}).get("llm_max_tokens", 220) or 220), 400)
        response = format_reply_for_tts(
            await brain.respond(
                system_prompt=system,
                transcript=state["transcript"],
                max_tokens=max_out,
                trace=trace,
            )
        )
        next_node = await _pick_next_node(
            brain=brain,
            state=state,
            current_id=node_id,
            outgoing=outgoing,
            trace=trace,
        )
        logger.info(
            "graph_agent_node node={} next={} latency_ms={:.0f}",
            node_id,
            next_node,
            (time.perf_counter() - t0) * 1000,
        )
        return _append_agent(state, response, node_id, next_node)

    return graph_agent_step


def adjacency_for_bot_config(bot_config: dict[str, Any]) -> dict[str, list[str]]:
    spec = parse_conversation_spec(bot_config.get("conversation_spec"))
    if spec is None or spec.mode != "graph":
        return {}
    return build_adjacency(spec)
