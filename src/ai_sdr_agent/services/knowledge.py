"""Per-node knowledge base retrieval for graph subagents.

Each graph node (and the single-mode agent) can have one or more knowledge
bases attached via ``agent_node_knowledge_bases``; if no per-node row exists we
fall back to bot-level ``bot_knowledge_bases``. We embed the user's latest
utterance with OpenAI and call the existing ``match_knowledge_chunks_for_user``
RPC, then format the top chunks as a system-prompt context block.

Latency budget: 1 OpenAI embedding round-trip (~120ms) + 1 SQL RPC (~30ms).
Failures are swallowed and return ``""`` so the conversation keeps flowing.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx
from loguru import logger
from sqlalchemy import text

from ai_sdr_agent.config import get_settings
from ai_sdr_agent.db.engine import get_async_session_factory

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_MODEL = DEFAULT_EMBEDDING_MODEL  # back-compat alias
EMBEDDING_DIM = 1536
DEFAULT_MATCH_COUNT = 5
DEFAULT_MIN_SIMILARITY = 0.2
DEFAULT_MAX_CONTEXT_CHARS = 6000
MAX_CONTEXT_CHARS = DEFAULT_MAX_CONTEXT_CHARS  # back-compat alias
CACHE_TTL_SECONDS = 300
KB_IDS_CACHE_TTL_SECONDS = 60
EMBEDDING_TIMEOUT_SECONDS = 6.0
RPC_TIMEOUT_SECONDS = 4.0
_MIN_TOKEN_LEN = 3
_MIN_ALPHA_TOKENS = 1


def resolve_kb_settings(bot_config: dict[str, Any] | None) -> dict[str, Any]:
    """Pull KB retrieval settings from a bot_config dict with hard defaults.

    Any field that's missing or out of range falls back to the module default.
    Returned dict is safe to splat into ``lookup_knowledge_chunks(**settings)``.
    """
    cfg = bot_config or {}

    def _clamp_int(value: Any, default: int, lo: int, hi: int) -> int:
        try:
            v = int(value) if value is not None else default
        except (TypeError, ValueError):
            v = default
        return max(lo, min(v, hi))

    def _clamp_float(value: Any, default: float, lo: float, hi: float) -> float:
        try:
            v = float(value) if value is not None else default
        except (TypeError, ValueError):
            v = default
        return max(lo, min(v, hi))

    def _coerce_str(value: Any, default: str) -> str:
        if not isinstance(value, str):
            return default
        cleaned = value.strip()
        return cleaned or default

    return {
        "match_count": _clamp_int(cfg.get("kb_match_count"), DEFAULT_MATCH_COUNT, 1, 20),
        "min_similarity": _clamp_float(
            cfg.get("kb_min_similarity"), DEFAULT_MIN_SIMILARITY, 0.0, 1.0
        ),
        "embedding_model": _coerce_str(cfg.get("kb_embedding_model"), DEFAULT_EMBEDDING_MODEL),
        "max_context_chars": _clamp_int(
            cfg.get("kb_max_context_chars"), DEFAULT_MAX_CONTEXT_CHARS, 500, 20000
        ),
        "max_tool_iterations": _clamp_int(cfg.get("kb_max_tool_iterations"), 2, 1, 4),
    }


@dataclass
class _CachedAnswer:
    expires_at: float
    context: str


@dataclass
class _CachedKbIds:
    expires_at: float
    kb_ids: list[str]


_cache: dict[tuple[str, str, str], _CachedAnswer] = {}
_kb_ids_cache: dict[tuple[str, str], _CachedKbIds] = {}
_CACHE_MAX = 1024


def _cache_get(key: tuple[str, str, str]) -> str | None:
    entry = _cache.get(key)
    if entry is None:
        return None
    if entry.expires_at < time.monotonic():
        _cache.pop(key, None)
        return None
    return entry.context


def _cache_put(key: tuple[str, str, str], context: str) -> None:
    if len(_cache) >= _CACHE_MAX:
        _cache.clear()
    _cache[key] = _CachedAnswer(time.monotonic() + CACHE_TTL_SECONDS, context)


def _question_is_trivial(question: str) -> bool:
    """Skip retrieval for short acknowledgements ("yes", "okay", "hm") where it adds no value."""
    tokens = re.findall(r"[^\W_]+", question, flags=re.UNICODE)
    meaningful = [t for t in tokens if len(t) >= _MIN_TOKEN_LEN]
    return len(meaningful) < _MIN_ALPHA_TOKENS or len(question.strip()) < 4


async def _resolve_kb_ids(*, bot_id: str, node_id: str | None, user_id: str) -> list[str]:
    """Per-node assignments first, otherwise fall back to bot-level (mirrors the edge function)."""
    try:
        session_factory = get_async_session_factory()
    except RuntimeError:
        logger.debug("kb:resolve skipped: db session factory not initialised")
        return []

    async with session_factory() as session:
        if node_id:
            node_rows = await session.execute(
                text(
                    "SELECT DISTINCT knowledge_base_id::text AS kb "
                    "FROM agent_node_knowledge_bases "
                    "WHERE bot_id = :bot AND node_id = :node AND user_id = :user"
                ),
                {"bot": bot_id, "node": node_id, "user": user_id},
            )
            ids = [row.kb for row in node_rows]
            if ids:
                return ids

        bot_rows = await session.execute(
            text(
                "SELECT DISTINCT knowledge_base_id::text AS kb "
                "FROM bot_knowledge_bases "
                "WHERE bot_id = :bot AND user_id = :user"
            ),
            {"bot": bot_id, "user": user_id},
        )
        return [row.kb for row in bot_rows]


async def list_kb_ids_for_node(
    *,
    bot_id: str | None,
    node_id: str | None,
    user_id: str | None,
) -> list[str]:
    """Cached lookup of KB ids attached to (bot, node), with bot-level fallback.

    Cached for ``KB_IDS_CACHE_TTL_SECONDS`` so repeated turn-time checks during a
    single voice call don't hammer the DB. Returns ``[]`` on any failure.
    """
    if not bot_id or not user_id:
        return []
    cache_key = (bot_id, node_id or "")
    cached = _kb_ids_cache.get(cache_key)
    now = time.monotonic()
    if cached is not None and cached.expires_at > now:
        return list(cached.kb_ids)
    try:
        ids = await _resolve_kb_ids(bot_id=bot_id, node_id=node_id, user_id=user_id)
    except Exception:
        logger.exception("kb:list_ids_failed bot=%s node=%s", bot_id, node_id or "-")
        ids = []
    if len(_kb_ids_cache) > _CACHE_MAX:
        _kb_ids_cache.clear()
    _kb_ids_cache[cache_key] = _CachedKbIds(now + KB_IDS_CACHE_TTL_SECONDS, list(ids))
    return list(ids)


def invalidate_kb_ids_cache(*, bot_id: str | None = None) -> None:
    """Clear the cached KB-id assignments (e.g. after the user changes wiring)."""
    if bot_id is None:
        _kb_ids_cache.clear()
        return
    for key in list(_kb_ids_cache.keys()):
        if key[0] == bot_id:
            _kb_ids_cache.pop(key, None)


async def _embed_question(
    question: str,
    *,
    openai_api_key: str,
    model: str = DEFAULT_EMBEDDING_MODEL,
) -> list[float] | None:
    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json",
    }
    payload = {"model": model, "input": question}

    async with httpx.AsyncClient(timeout=EMBEDDING_TIMEOUT_SECONDS) as client:
        resp = await client.post(
            "https://api.openai.com/v1/embeddings",
            json=payload,
            headers=headers,
        )

    if resp.status_code != 200:
        logger.warning(
            "kb:embedding_failed status={} body={}",
            resp.status_code,
            resp.text[:200],
        )
        return None

    data = resp.json()
    vec = (data.get("data") or [{}])[0].get("embedding")
    if not isinstance(vec, list) or len(vec) != EMBEDDING_DIM:
        logger.warning(
            "kb:embedding_unexpected_shape len={} expected={}",
            len(vec) if isinstance(vec, list) else None,
            EMBEDDING_DIM,
        )
        return None
    return vec


async def _match_chunks(
    *,
    embedding: list[float],
    user_id: str,
    kb_ids: list[str],
    match_count: int,
) -> list[dict[str, Any]]:
    try:
        session_factory = get_async_session_factory()
    except RuntimeError:
        return []

    vector_literal = "[" + ",".join(f"{x:.6f}" for x in embedding) + "]"

    async with session_factory() as session:
        result = await asyncio.wait_for(
            session.execute(
                text(
                    "SELECT id::text AS id, knowledge_base_id::text AS knowledge_base_id, "
                    "document_id::text AS document_id, chunk_index, content, similarity "
                    "FROM match_knowledge_chunks_for_user("
                    "CAST(:emb AS vector(1536)), CAST(:user AS uuid), :n, "
                    "CAST(:kbs AS uuid[]))"
                ),
                {
                    "emb": vector_literal,
                    "user": user_id,
                    "n": match_count,
                    "kbs": kb_ids,
                },
            ),
            timeout=RPC_TIMEOUT_SECONDS,
        )
        return [dict(row) for row in result.mappings()]


def _format_context(
    matches: list[dict[str, Any]],
    *,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
) -> str:
    if not matches:
        return ""

    parts: list[str] = []
    total = 0
    for i, m in enumerate(matches, start=1):
        content = (m.get("content") or "").strip()
        if not content:
            continue
        block = f"[{i}] {content}"
        if total + len(block) > max_context_chars:
            break
        parts.append(block)
        total += len(block)

    if not parts:
        return ""

    return (
        "Knowledge base context (use only this for any factual claims; cite as [n] if asked, "
        "do not invent beyond what is here):\n" + "\n\n".join(parts)
    )


def _format_tool_result(
    matches: list[dict[str, Any]],
    *,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
) -> str:
    """Compact, header-less rendering for use as an LLM tool result string."""
    if not matches:
        return "NO_RESULTS"

    parts: list[str] = []
    total = 0
    for i, m in enumerate(matches, start=1):
        content = (m.get("content") or "").strip()
        if not content:
            continue
        sim = m.get("similarity")
        try:
            sim_str = f" (similarity={float(sim):.2f})" if sim is not None else ""
        except (TypeError, ValueError):
            sim_str = ""
        block = f"[{i}]{sim_str} {content}"
        if total + len(block) > max_context_chars:
            break
        parts.append(block)
        total += len(block)

    if not parts:
        return "NO_RESULTS"
    return "\n\n".join(parts)


async def lookup_knowledge_chunks(
    *,
    bot_id: str | None,
    node_id: str | None,
    user_id: str | None,
    question: str,
    openai_api_key: str | None = None,
    kb_ids: list[str] | None = None,
    match_count: int = DEFAULT_MATCH_COUNT,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
) -> str:
    """Tool-style retrieval that returns just the formatted chunks (or NO_RESULTS).

    Used by ``ConversationBrain.respond_with_tools`` as the executor for the
    ``lookup_knowledge`` function. Best-effort: errors/timeouts return a short
    error sentinel so the model can recover gracefully.

    Settings ``match_count``, ``min_similarity``, ``embedding_model`` and
    ``max_context_chars`` are normally derived per-bot from
    ``bot_configs.kb_*`` via :func:`resolve_kb_settings`.
    """
    q = (question or "").strip()
    if not q:
        return "NO_RESULTS"

    try:
        resolved_kb_ids = (
            kb_ids
            if kb_ids
            else await list_kb_ids_for_node(bot_id=bot_id, node_id=node_id, user_id=user_id)
        )
        if not resolved_kb_ids or not user_id:
            return "NO_RESULTS"

        key = openai_api_key or get_settings().openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            logger.warning(
                "kb_tool:skip_no_openai_key bot={} node={}",
                bot_id,
                node_id or "-",
            )
            return "NO_KNOWLEDGE_BACKEND_CONFIGURED"

        t0 = time.perf_counter()
        embedding = await _embed_question(q, openai_api_key=key, model=embedding_model)
        if embedding is None:
            return "NO_RESULTS"

        matches = await _match_chunks(
            embedding=embedding,
            user_id=user_id,
            kb_ids=resolved_kb_ids,
            match_count=match_count,
        )
        usable = [m for m in matches if (m.get("similarity") or 0) >= min_similarity]
        rendered = _format_tool_result(usable, max_context_chars=max_context_chars)

        logger.info(
            "kb_tool:executed bot={} node={} kb_count={} q_chars={} matches={} usable={} "
            "top_sim={} result_chars={} model={} min_sim={:.2f} latency_ms={:.0f}",
            bot_id,
            node_id or "-",
            len(resolved_kb_ids),
            len(q),
            len(matches),
            len(usable),
            f"{matches[0]['similarity']:.3f}" if matches else "-",
            len(rendered),
            embedding_model,
            min_similarity,
            (time.perf_counter() - t0) * 1000,
        )
        return rendered
    except asyncio.CancelledError:
        raise
    except asyncio.TimeoutError:
        logger.warning("kb_tool:timeout bot={} node={}", bot_id, node_id or "-")
        return "ERROR: knowledge base lookup timed out"
    except Exception:
        logger.exception("kb_tool:error bot=%s node=%s", bot_id, node_id or "-")
        return "ERROR: knowledge base lookup failed"


async def retrieve_node_context(
    *,
    bot_id: str | None,
    node_id: str | None,
    user_id: str | None,
    question: str,
    openai_api_key: str | None = None,
    match_count: int = DEFAULT_MATCH_COUNT,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
) -> str:
    """Return a formatted KB context block, or ``""`` when nothing usable was found.

    Best-effort: any error/timeout returns ``""`` so the conversation continues
    even when the KB is misconfigured.
    """
    q = (question or "").strip()
    if not q or not bot_id or not user_id:
        return ""
    if _question_is_trivial(q):
        return ""

    cache_key = (bot_id, node_id or "", q.lower())
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    try:
        kb_ids = await _resolve_kb_ids(bot_id=bot_id, node_id=node_id, user_id=user_id)
        if not kb_ids:
            _cache_put(cache_key, "")
            logger.debug(
                "kb:no_assignments bot={} node={}",
                bot_id,
                node_id or "-",
            )
            return ""

        key = openai_api_key or get_settings().openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            logger.warning(
                "kb:skip_no_openai_key bot={} node={} (set OPENAI_API_KEY or bot.openai_api_key to enable RAG)",
                bot_id,
                node_id or "-",
            )
            return ""

        t0 = time.perf_counter()
        embedding = await _embed_question(q, openai_api_key=key)
        if embedding is None:
            return ""

        matches = await _match_chunks(
            embedding=embedding,
            user_id=user_id,
            kb_ids=kb_ids,
            match_count=match_count,
        )
        usable = [m for m in matches if (m.get("similarity") or 0) >= min_similarity]
        context = _format_context(usable)

        logger.info(
            "kb:retrieved bot={} node={} kb_count={} matches={} usable={} top_sim={} "
            "context_chars={} latency_ms={:.0f}",
            bot_id,
            node_id or "-",
            len(kb_ids),
            len(matches),
            len(usable),
            f"{matches[0]['similarity']:.3f}" if matches else "-",
            len(context),
            (time.perf_counter() - t0) * 1000,
        )

        _cache_put(cache_key, context)
        return context
    except asyncio.CancelledError:
        raise
    except asyncio.TimeoutError:
        logger.warning("kb:timeout bot={} node={}", bot_id, node_id or "-")
        return ""
    except Exception:
        logger.exception("kb:error bot=%s node=%s (best-effort, returning empty)", bot_id, node_id or "-")
        return ""
