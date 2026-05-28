from __future__ import annotations

from typing import Any

from ai_sdr_agent.services.tool_config import HttpToolConfigV1, ToolParameterDef


def _param_property(defn: ToolParameterDef) -> dict[str, Any]:
    prop: dict[str, Any] = {"type": defn.type, "description": defn.description or defn.name}
    return prop


def build_parameters_schema(config: HttpToolConfigV1) -> dict[str, Any]:
    if config.parameters and isinstance(config.parameters, dict):
        schema = dict(config.parameters)
        if schema.get("type") != "object":
            schema = {"type": "object", "properties": schema.get("properties", {})}
        if "additionalProperties" not in schema:
            schema["additionalProperties"] = False
        return schema

    properties: dict[str, Any] = {}
    required: list[str] = []
    for defn in (
        list(config.path_parameters)
        + list(config.query_parameters)
    ):
        properties[defn.name] = _param_property(defn)
        if defn.required:
            required.append(defn.name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }
