from __future__ import annotations

import ipaddress
import json
import socket
from typing import Any, Mapping
from urllib.parse import urlparse

import httpx
from loguru import logger

from ai_sdr_agent.db.models import AuthConnectionRow
from ai_sdr_agent.services.env_substitution import substitute_env_vars
from ai_sdr_agent.services.tool_config import HttpToolConfigV1, ToolAuthConfig, parse_tool_config

MAX_RESPONSE_CHARS = 16_000


def _is_blocked_host(host: str) -> bool:
    if not host:
        return True
    lowered = host.lower().strip()
    if lowered in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
        return True
    try:
        addr = ipaddress.ip_address(lowered)
        return addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved
    except ValueError:
        pass
    try:
        for info in socket.getaddrinfo(host, None):
            ip = info[4][0]
            parsed = ipaddress.ip_address(ip)
            if parsed.is_private or parsed.is_loopback or parsed.is_link_local:
                return True
    except OSError:
        return True
    return False


def validate_url_ssrf(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("Only http and https URLs are allowed")
    if _is_blocked_host(parsed.hostname or ""):
        raise ValueError("URL host is not allowed")


def _apply_path_params(url_template: str, args: dict[str, Any]) -> str:
    result = url_template
    for key, value in args.items():
        placeholder = "{" + key + "}"
        if placeholder in result:
            result = result.replace(placeholder, str(value))
    return result


def _build_query_params(config: HttpToolConfigV1, args: dict[str, Any]) -> dict[str, str]:
    names = {p.name for p in config.query_parameters}
    return {k: str(v) for k, v in args.items() if k in names and v is not None}


def _resolve_auth_headers(
    auth: ToolAuthConfig,
    env: Mapping[str, str],
    connection: AuthConnectionRow | None = None,
    *,
    connection_config: dict[str, Any] | None = None,
    connection_type: str | None = None,
) -> dict[str, str]:
    headers: dict[str, str] = {}
    auth_type = auth.type
    cfg = connection_config or (connection.config_json if connection else {})

    if auth_type == "connection":
        auth_type = connection_type or cfg.get("type", "api_key_header")

    if auth_type == "bearer":
        token = auth.bearer_token or cfg.get("bearer_token") or ""
        token = substitute_env_vars(token, env) if token else ""
        if token:
            headers["Authorization"] = f"Bearer {token}"
    elif auth_type == "basic":
        user = auth.basic_username or cfg.get("username") or ""
        password = auth.basic_password or cfg.get("password") or ""
        user = substitute_env_vars(user, env) if user else ""
        password = substitute_env_vars(password, env) if password else ""
        if user or password:
            import base64

            cred = base64.b64encode(f"{user}:{password}".encode()).decode()
            headers["Authorization"] = f"Basic {cred}"
    elif auth_type == "api_key_header":
        header_name = auth.api_key_header_name or cfg.get("header_name") or "X-Api-Key"
        value = auth.api_key_value or cfg.get("api_key") or ""
        value = substitute_env_vars(value, env) if value else ""
        if value:
            headers[header_name] = value
    return headers


async def execute_http_tool(
    *,
    config: HttpToolConfigV1,
    args: dict[str, Any],
    env: Mapping[str, str],
    connection: AuthConnectionRow | None = None,
    connection_config: dict[str, Any] | None = None,
    connection_type: str | None = None,
    log_context: str = "-",
) -> str:
    url = substitute_env_vars(_apply_path_params(config.url, args), env)
    validate_url_ssrf(url)

    headers: dict[str, str] = {}
    headers.update(
        _resolve_auth_headers(
            config.auth,
            env,
            connection,
            connection_config=connection_config,
            connection_type=connection_type,
        )
    )
    for h in config.headers:
        if h.name.strip():
            headers[h.name.strip()] = substitute_env_vars(h.value, env)

    query = _build_query_params(config, args)
    body_keys = set(args.keys()) - {p.name for p in config.path_parameters} - {p.name for p in config.query_parameters}
    json_body = {k: args[k] for k in body_keys if k in args} if body_keys else None

    timeout = float(config.response_timeout_seconds)
    method = config.method.upper()

    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
            response = await client.request(
                method,
                url,
                headers=headers,
                params=query or None,
                json=json_body if method in ("POST", "PUT", "PATCH") and json_body else None,
            )
    except httpx.TimeoutException:
        logger.warning("http_tool:timeout context={} url={}", log_context, url[:80])
        return "ERROR: HTTP request timed out"
    except Exception as exc:
        logger.warning("http_tool:error context={} err={}", log_context, exc)
        return f"ERROR: HTTP request failed: {exc}"

    text = response.text[:MAX_RESPONSE_CHARS]
    if len(response.text) > MAX_RESPONSE_CHARS:
        text += "\n...(truncated)"

    logger.info(
        "http_tool:done context={} status={} chars={}",
        log_context,
        response.status_code,
        len(text),
    )

    if response.status_code >= 400:
        return f"ERROR: HTTP {response.status_code}\n{text}"

    content_type = (response.headers.get("content-type") or "").lower()
    if "application/json" in content_type:
        try:
            return json.dumps(response.json(), ensure_ascii=False)[:MAX_RESPONSE_CHARS]
        except json.JSONDecodeError:
            pass
    return text or "(empty response)"


def tool_row_to_runtime_dict(row: Any) -> dict[str, Any]:
    config = parse_tool_config(row.config_json if hasattr(row, "config_json") else {})
    return {
        "id": str(row.id),
        "name": row.name,
        "description": row.description or "",
        "kind": row.kind,
        "config": config.model_dump(),
    }
