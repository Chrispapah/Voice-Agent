from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from ai_sdr_agent.db.repositories import (
    PgAgentToolRepository,
    PgAuthConnectionRepository,
    PgWorkspaceEnvVarRepository,
)
from ai_sdr_agent.services.tool_config import parse_tool_config
from ai_sdr_agent.graph.spec import collect_tool_ids_from_spec, parse_conversation_spec
from ai_sdr_agent.services.http_tool_executor import tool_row_to_runtime_dict


async def preload_agent_tools_for_bot(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    bot_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    spec_raw = bot_config.get("conversation_spec")
    spec = parse_conversation_spec(spec_raw) if spec_raw else None
    tool_ids = collect_tool_ids_from_spec(spec) if spec else []
    env_repo = PgWorkspaceEnvVarRepository(session)
    env_rows = await env_repo.list_for_user(user_id)
    env_map = {r.name: r.value for r in env_rows}
    if not tool_ids:
        return [], env_map

    tool_repo = PgAgentToolRepository(session)
    rows = await tool_repo.list_by_ids(user_id=user_id, tool_ids=tool_ids)
    auth_repo = PgAuthConnectionRepository(session)
    cache: list[dict[str, Any]] = []
    for r in rows:
        entry = tool_row_to_runtime_dict(r)
        config = parse_tool_config(r.config_json)
        if config.auth.type == "connection" and config.auth.connection_id:
            try:
                conn = await auth_repo.get(uuid.UUID(config.auth.connection_id), user_id)
                if conn:
                    entry["auth_connection_config"] = conn.config_json
                    entry["auth_connection_type"] = conn.type
            except (ValueError, TypeError):
                pass
        cache.append(entry)
    return cache, env_map


def merge_tools_cache_into_state(state: dict[str, Any], cache: list[dict], env_map: dict[str, str]) -> None:
    meta = state.setdefault("metadata", {})
    meta["agent_tools_cache"] = cache
    meta["workspace_env"] = env_map
