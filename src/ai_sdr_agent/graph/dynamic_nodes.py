from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from loguru import logger

from ai_sdr_agent.graph.prompts import _VOICE_OUTPUT_RULES, _template_vars, format_reply_for_tts
from ai_sdr_agent.graph.spec import (
    SINGLE_AGENT_NODE_ID,
    ReplyTurnMode,
    build_adjacency,
    parse_conversation_spec,
    prompt_for_node,
    reply_turn_modes_for_node,
    static_message_for_node,
)
from ai_sdr_agent.graph.state import ConversationState

if TYPE_CHECKING:
    from ai_sdr_agent.services.brain import ConversationBrain

# One warning per (conversation_id, graph node) — avoids log spam while a session is stuck.
_sticky_routing_warned: set[tuple[str, str]] = set()
_STICKY_WARN_CAP = 4096


def _pick_reply_mode(
    *,
    modes: list[ReplyTurnMode] | None,
    utterance_index: int,
    has_human: bool,
    static_text: str | None,
) -> ReplyTurnMode:
    """Resolve static vs LLM for this utterance.

    When ``modes`` is non-empty, index maps to successive agent lines at this node (0 = opener).
    Indices past the list default to ``llm``. ``static`` falls back to ``llm`` if ``static_text`` is empty.

    When ``modes`` is absent or empty, legacy graph behavior: after the user has spoken, a non-empty
    ``static_text`` forces ``static``; the opener uses ``llm``.
    """
    trimmed = (static_text or "").strip()
    if modes:
        mode: ReplyTurnMode = modes[utterance_index] if utterance_index < len(modes) else "llm"
        if mode == "static" and not trimmed:
            logger.warning(
                "reply_turn_modes requested static but static_message is empty; using llm instead "
                "(utterance_index={})",
                utterance_index,
            )
            return "llm"
        return mode
    if has_human and trimmed:
        return "static"
    return "llm"


def _maybe_warn_sticky_routing(state: ConversationState, node_id: str, outgoing: list[str]) -> None:
    """If this node cannot leave (0 outgoing or only a self-loop), classify never advances the flow."""
    trapped = (not outgoing) or (len(outgoing) == 1 and outgoing[0] == node_id)
    if not trapped:
        return
    meta = state.get("metadata") or {}
    conv = str(meta.get("conversation_id", "") or "").strip()
    if not conv:
        return
    key = (conv, node_id)
    if key in _sticky_routing_warned:
        return
    if len(_sticky_routing_warned) >= _STICKY_WARN_CAP:
        _sticky_routing_warned.clear()
    _sticky_routing_warned.add(key)
    logger.warning(
        "Graph node {!r} cannot advance (outgoing_edges={}). Add outbound edges to the next "
        "stage or to 'complete'. Otherwise turns stay here until max_call_turns triggers goodbye.",
        node_id,
        outgoing,
    )


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


def _classify_routing_context(
    spec: ConversationSpecV1,
    *,
    current_id: str,
    outgoing: list[str],
) -> str:
    """Human-readable destination lines + optional per-node hint for graph edge selection."""
    lines: list[str] = []
    for lid in outgoing:
        node = next((n for n in spec.nodes if n.id == lid), None)
        lbl = (node.label or "").strip() if node else ""
        lines.append(f"- {lid}: {lbl}" if lbl else f"- {lid}")
    block = "\n".join(lines)
    cur = next((n for n in spec.nodes if n.id == current_id), None)
    hint = (cur.classify_hint or "").strip() if cur else ""
    parts = [
        "",
        "Candidate destinations (respond with exactly one id from your allowed list):",
        block,
    ]
    if hint:
        parts.extend(["", "Routing guidance:", hint])
    return "\n".join(parts)


async def _pick_next_node(
    *,
    brain: ConversationBrain,
    state: ConversationState,
    spec: ConversationSpecV1,
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
    instruction += _classify_routing_context(spec, current_id=current_id, outgoing=outgoing)
    choice = await brain.classify(
        instruction=instruction,
        human_input=human,
        labels=outgoing,
        trace=trace,
    )
    if choice in outgoing:
        return choice
    return outgoing[0]


def _loop_limits_for_node(spec: ConversationSpecV1, node_id: str) -> tuple[int | None, int | None]:
    node = next((n for n in spec.nodes if n.id == node_id), None)
    if node is None:
        return None, None
    lo = node.loop_min_turns
    hi = node.loop_max_turns
    if lo is not None and lo < 1:
        lo = None
    return lo, hi


def _apply_loop_min_max(
    *,
    node_id: str,
    outgoing: list[str],
    raw_next: str,
    prior_streak: int,
    loop_min: int | None,
    loop_max: int | None,
) -> str:
    has_self = node_id in outgoing
    non_self = next((t for t in outgoing if t != node_id), None)
    next_n = raw_next if raw_next in outgoing else outgoing[0]
    if loop_max is not None and non_self is not None and prior_streak >= loop_max:
        if next_n == node_id:
            return non_self
    if loop_min is not None and has_self and non_self is not None:
        if next_n != node_id and prior_streak < loop_min:
            return node_id
    return next_n


def _append_agent(
    state: ConversationState,
    content: str,
    node_name: str,
    next_node: str,
    *,
    graph_node_streaks: dict[str, int] | None = None,
    graph_node_utterance_index: dict[str, int] | None = None,
) -> dict[str, Any]:
    streaks: dict[str, int] = dict(
        graph_node_streaks
        if graph_node_streaks is not None
        else (state.get("graph_node_streaks") or {})
    )
    utter_idx: dict[str, int] = dict(
        graph_node_utterance_index
        if graph_node_utterance_index is not None
        else (state.get("graph_node_utterance_index") or {})
    )
    out: dict[str, Any] = {
        "transcript": state["transcript"] + [{"role": "agent", "content": content}],
        "current_node": node_name,
        "last_agent_response": content,
        "next_node": next_node,
        "route_decision": next_node,
        "graph_node_streaks": streaks,
        "graph_node_utterance_index": utter_idx,
    }
    return out


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
        utter_map = dict(state.get("graph_node_utterance_index") or {})
        idx = int(utter_map.get(SINGLE_AGENT_NODE_ID, 0))
        static_src = (spec.single_static_message or "").strip() or None
        modes = spec.single_reply_turn_modes
        mode = _pick_reply_mode(
            modes=modes,
            utterance_index=idx,
            has_human=has_human,
            static_text=static_src,
        )
        if mode == "static":
            text = format_reply_for_tts(_interpolate_placeholders(static_src or "", state))
            logger.info(
                "single_agent_node static_message latency_ms={:.0f}",
                (time.perf_counter() - t0) * 1000,
            )
        else:
            system = _interpolate_placeholders(spec.system_prompt or "", state)
            system = f"{system.strip()}\n{_VOICE_OUTPUT_RULES}"
            max_out = min(int(state.get("bot_config", {}).get("llm_max_tokens", 220) or 220), 400)
            text = format_reply_for_tts(
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
        next_n = SINGLE_AGENT_NODE_ID
        new_um = dict(utter_map)
        new_um[SINGLE_AGENT_NODE_ID] = idx + 1
        return _append_agent(
            state,
            text,
            SINGLE_AGENT_NODE_ID,
            next_n,
            graph_node_utterance_index=new_um,
        )

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
        _maybe_warn_sticky_routing(state, node_id, outgoing)

        utter_map = dict(state.get("graph_node_utterance_index") or {})

        if not has_human:
            utter_idx = int(utter_map.get(node_id, 0))
            modes = reply_turn_modes_for_node(spec, node_id)
            static_src = static_message_for_node(spec, node_id)
            reply_kind = _pick_reply_mode(
                modes=modes,
                utterance_index=utter_idx,
                has_human=False,
                static_text=static_src,
            )
            if reply_kind == "static":
                response = format_reply_for_tts(_interpolate_placeholders(static_src or "", state))
                logger.info(
                    "graph_agent_node opener static node={} latency_ms={:.0f}",
                    node_id,
                    (time.perf_counter() - t0) * 1000,
                )
            else:
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
                logger.info(
                    "graph_agent_node opener llm node={} latency_ms={:.0f}",
                    node_id,
                    (time.perf_counter() - t0) * 1000,
                )
            if len(outgoing) == 1:
                next_node = outgoing[0]
            else:
                next_node = node_id
            streaks = dict(state.get("graph_node_streaks") or {})
            prior = int(streaks.get(node_id, 0))
            if len(outgoing) > 1:
                loop_min, loop_max = _loop_limits_for_node(spec, node_id)
                next_node = _apply_loop_min_max(
                    node_id=node_id,
                    outgoing=outgoing,
                    raw_next=next_node,
                    prior_streak=prior,
                    loop_min=loop_min,
                    loop_max=loop_max,
                )
            if next_node == node_id:
                streaks[node_id] = prior + 1
            else:
                streaks[node_id] = 0
            new_um = dict(utter_map)
            if next_node == node_id:
                new_um[node_id] = utter_idx + 1
            else:
                new_um.pop(node_id, None)
            return _append_agent(
                state,
                response,
                node_id,
                next_node,
                graph_node_streaks=streaks,
                graph_node_utterance_index=new_um,
            )

        raw_next = await _pick_next_node(
            brain=brain,
            state=state,
            spec=spec,
            current_id=node_id,
            outgoing=outgoing,
            trace=trace,
        )
        streaks = dict(state.get("graph_node_streaks") or {})
        prior = int(streaks.get(node_id, 0))
        next_node = raw_next
        if len(outgoing) > 1:
            loop_min, loop_max = _loop_limits_for_node(spec, node_id)
            next_node = _apply_loop_min_max(
                node_id=node_id,
                outgoing=outgoing,
                raw_next=raw_next,
                prior_streak=prior,
                loop_min=loop_min,
                loop_max=loop_max,
            )

        speak_node = next_node
        if next_node == "complete":
            speak_node = node_id

        speak_idx = int(utter_map.get(speak_node, 0))
        speak_modes = reply_turn_modes_for_node(spec, speak_node)
        speak_static = static_message_for_node(spec, speak_node)
        reply_kind = _pick_reply_mode(
            modes=speak_modes,
            utterance_index=speak_idx,
            has_human=True,
            static_text=speak_static,
        )

        trace_reply = {**_trace(state), "node": speak_node}
        if reply_kind == "static":
            response = format_reply_for_tts(_interpolate_placeholders(speak_static or "", state))
            logger.info(
                "graph_agent_node static_message route_from={} speak_as={} latency_ms={:.0f}",
                node_id,
                speak_node,
                (time.perf_counter() - t0) * 1000,
            )
        else:
            base_prompt = _interpolate_placeholders(prompt_for_node(spec, speak_node), state)
            system = f"{base_prompt.strip()}\n{_VOICE_OUTPUT_RULES}"
            max_out = min(int(state.get("bot_config", {}).get("llm_max_tokens", 220) or 220), 400)
            response = format_reply_for_tts(
                await brain.respond(
                    system_prompt=system,
                    transcript=state["transcript"],
                    max_tokens=max_out,
                    trace=trace_reply,
                )
            )

        if next_node == node_id:
            streaks[node_id] = prior + 1
        else:
            streaks[node_id] = 0

        new_um = dict(utter_map)
        new_um[speak_node] = speak_idx + 1
        if node_id != next_node:
            new_um.pop(node_id, None)

        logger.info(
            "graph_agent_node route_from={} speak_as={} next={} latency_ms={:.0f}",
            node_id,
            speak_node,
            next_node,
            (time.perf_counter() - t0) * 1000,
        )
        return _append_agent(
            state,
            response,
            speak_node,
            next_node,
            graph_node_streaks=streaks,
            graph_node_utterance_index=new_um,
        )

    return graph_agent_step


def adjacency_for_bot_config(bot_config: dict[str, Any]) -> dict[str, list[str]]:
    spec = parse_conversation_spec(bot_config.get("conversation_spec"))
    if spec is None or spec.mode != "graph":
        return {}
    return build_adjacency(spec)
