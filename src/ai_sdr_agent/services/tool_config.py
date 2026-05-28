from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

HttpMethod = Literal["GET", "POST", "PUT", "PATCH", "DELETE"]
PreToolSpeech = Literal["auto", "force", "disabled"]
ExecutionMode = Literal["default", "blocking"]
ToolCallSound = Literal["none", "click", "custom_url"]
AuthType = Literal["none", "bearer", "basic", "api_key_header", "connection"]


class ToolHeader(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    value: str = ""


class ToolParameterDef(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    description: str = ""
    type: Literal["string", "number", "integer", "boolean"] = "string"
    required: bool = False


class ToolAuthConfig(BaseModel):
    type: AuthType = "none"
    bearer_token: str | None = None
    basic_username: str | None = None
    basic_password: str | None = None
    api_key_header_name: str | None = None
    api_key_value: str | None = None
    connection_id: str | None = None


class HttpToolConfigV1(BaseModel):
    schema_version: Literal[1] = 1
    method: HttpMethod = "POST"
    url: str = ""
    response_timeout_seconds: int = Field(default=20, ge=1, le=120)
    disable_interruptions: bool = False
    pre_tool_speech: PreToolSpeech = "auto"
    pre_tool_speech_text: str | None = None
    execution_mode: ExecutionMode = "default"
    tool_call_sound: ToolCallSound = "none"
    tool_call_sound_url: str | None = None
    auth: ToolAuthConfig = Field(default_factory=ToolAuthConfig)
    headers: list[ToolHeader] = Field(default_factory=list)
    path_parameters: list[ToolParameterDef] = Field(default_factory=list)
    query_parameters: list[ToolParameterDef] = Field(default_factory=list)
    parameters: dict[str, Any] | None = None

    @field_validator("url")
    @classmethod
    def url_not_empty_when_saved(cls, v: str) -> str:
        return v.strip()


_ENV_PATTERN = re.compile(r"\{\{([A-Za-z_][A-Za-z0-9_]*)\}\}")


def extract_path_param_names(url: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", url)))


def parse_tool_config(raw: dict[str, Any] | None) -> HttpToolConfigV1:
    if not raw:
        return HttpToolConfigV1()
    data = dict(raw)
    if data.get("schema_version") != 1:
        migrated = _migrate_legacy_config(data)
        return HttpToolConfigV1.model_validate(migrated)
    return HttpToolConfigV1.model_validate(data)


def _migrate_legacy_config(data: dict[str, Any]) -> dict[str, Any]:
    url = data.get("url") or data.get("endpoint_url") or ""
    method = (data.get("method") or "POST").upper()
    if method not in ("GET", "POST", "PUT", "PATCH", "DELETE"):
        method = "POST"
    return {
        "schema_version": 1,
        "method": method,
        "url": url,
        "response_timeout_seconds": 20,
    }


def llm_function_name(tool_id: str, display_name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_]+", "_", display_name.strip().lower()).strip("_")
    if not slug:
        slug = "tool"
    short_id = str(tool_id).replace("-", "")[:8]
    name = f"tool_{short_id}_{slug}"
    if len(name) > 64:
        name = name[:64].rstrip("_")
    return name
