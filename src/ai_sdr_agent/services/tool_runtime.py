from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger

from ai_sdr_agent.graph.spec import tool_ids_for_node
from ai_sdr_agent.services.brain import ToolDefinition, get_response_chunk_sink
from ai_sdr_agent.services.http_tool_executor import execute_http_tool
from ai_sdr_agent.services.tool_config import HttpToolConfigV1, llm_function_name, parse_tool_config
from ai_sdr_agent.services.tool_context import (
    ActiveToolVoiceFlags,
    emit_tool_sound,
    reset_active_tool_voice,
    set_active_tool_voice,
)
from ai_sdr_agent.services.tool_schema import build_parameters_schema

ToolExecutorFn = Callable[[str, dict[str, Any]], Awaitable[str]]

_MAX_GLOBAL_TOOL_CALLS = 5


def _get_cache(state: dict[str, Any]) -> list[dict[str, Any]]:
    meta = state.get("metadata") or {}
    return list(meta.get("agent_tools_cache") or [])


def _get_env(state: dict[str, Any]) -> dict[str, str]:
    meta = state.get("metadata") or {}
    return dict(meta.get("workspace_env") or {})


def build_http_tool_definitions(
    tool_rows: list[dict[str, Any]],
    *,
    node_tool_ids: list[str],
) -> tuple[list[ToolDefinition], dict[str, dict[str, Any]]]:
    allowed = set(node_tool_ids)
    tools: list[ToolDefinition] = []
    by_fn: dict[str, dict[str, Any]] = {}
    for row in tool_rows:
        tid = row["id"]
        if allowed and tid not in allowed:
            continue
        if row.get("kind") == "custom":
            continue
        config = parse_tool_config(row.get("config") or {})
        if not config.url.strip():
            continue
        fn_name = llm_function_name(tid, row["name"])
        tools.append(
            ToolDefinition(
                name=fn_name,
                description=row.get("description") or row.get("name") or "HTTP tool",
                parameters=build_parameters_schema(config),
            )
        )
        by_fn[fn_name] = {**row, "config_parsed": config}
    return tools, by_fn


async def _maybe_pre_tool_speech(config: HttpToolConfigV1) -> None:
    if config.pre_tool_speech == "disabled":
        return
    if config.pre_tool_speech != "force":
        return
    text = (config.pre_tool_speech_text or "").strip() or "One moment while I check that."
    sink = get_response_chunk_sink()
    if sink is not None:
        await sink(text + " ")


async def _maybe_tool_sound(config: HttpToolConfigV1) -> None:
    if config.tool_call_sound == "none":
        return
    payload: dict[str, Any] = {"sound": config.tool_call_sound}
    if config.tool_call_sound == "custom_url" and config.tool_call_sound_url:
        payload["url"] = config.tool_call_sound_url
    await emit_tool_sound(payload)


def build_tool_executor(
    *,
    state: dict[str, Any],
    by_fn: dict[str, dict[str, Any]],
    kb_executor: ToolExecutorFn | None,
    call_budget: list[int] | None = None,
) -> ToolExecutorFn:
    env = _get_env(state)
    budget = call_budget if call_budget is not None else [_MAX_GLOBAL_TOOL_CALLS]

    async def executor(name: str, args: dict[str, Any]) -> str:
        if budget[0] <= 0:
            return "ERROR: tool call limit reached for this turn"
        budget[0] -= 1

        if kb_executor and name == "lookup_knowledge":
            return await kb_executor(name, args)

        row = by_fn.get(name)
        if not row:
            return f"ERROR: unknown tool {name!r}"

        config: HttpToolConfigV1 = row["config_parsed"]
        token = set_active_tool_voice(
            ActiveToolVoiceFlags(
                disable_interruptions=config.disable_interruptions,
                pre_tool_speech=config.pre_tool_speech,
                pre_tool_speech_text=config.pre_tool_speech_text,
                tool_call_sound=config.tool_call_sound,
                tool_call_sound_url=config.tool_call_sound_url,
            )
        )
        try:
            await _maybe_pre_tool_speech(config)
            await _maybe_tool_sound(config)
            return await execute_http_tool(
                config=config,
                args=args,
                env=env,
                connection_config=row.get("auth_connection_config"),
                connection_type=row.get("auth_connection_type"),
                log_context=str((state.get("metadata") or {}).get("conversation_id", "-")),
            )
        finally:
            reset_active_tool_voice(token)

    return executor


def resolve_node_tool_ids(state: dict[str, Any], node_id: str) -> list[str]:
    bot_cfg = state.get("bot_config") or {}
    spec_raw = bot_cfg.get("conversation_spec")
    if not spec_raw:
        return []
    from ai_sdr_agent.graph.spec import parse_conversation_spec

    spec = parse_conversation_spec(spec_raw)
    if spec is None:
        return []
    return tool_ids_for_node(spec, node_id)


def build_tooling_for_node(
    *,
    state: dict[str, Any],
    node_id: str,
    kb_tool: ToolDefinition | None,
) -> tuple[list[ToolDefinition], dict[str, dict[str, Any]]]:
    cache = _get_cache(state)
    node_ids = resolve_node_tool_ids(state, node_id)
    http_tools, by_fn = build_http_tool_definitions(cache, node_tool_ids=node_ids)
    tools: list[ToolDefinition] = []
    if kb_tool is not None:
        tools.append(kb_tool)
    tools.extend(http_tools)
    return tools, by_fn
