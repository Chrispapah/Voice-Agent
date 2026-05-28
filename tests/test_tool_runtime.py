from __future__ import annotations

import pytest

from ai_sdr_agent.graph.spec import collect_tool_ids_from_spec, tool_ids_for_node, ConversationSpecV1, SpecNode
from ai_sdr_agent.services.env_substitution import substitute_env_vars
from ai_sdr_agent.services.http_tool_executor import validate_url_ssrf
from ai_sdr_agent.services.tool_config import llm_function_name, parse_tool_config
from ai_sdr_agent.services.tool_runtime import build_http_tool_definitions
from ai_sdr_agent.services.tool_schema import build_parameters_schema


def test_parse_legacy_config_migrates_endpoint_url():
    cfg = parse_tool_config({"endpoint_url": "https://example.com/hook", "method": "get"})
    assert cfg.url == "https://example.com/hook"
    assert cfg.method == "GET"
    assert cfg.schema_version == 1


def test_substitute_env_vars():
    out = substitute_env_vars("https://x.com/{{API_KEY}}/path", {"API_KEY": "secret"})
    assert out == "https://x.com/secret/path"


def test_substitute_missing_var_raises():
    with pytest.raises(KeyError, match="Missing"):
        substitute_env_vars("{{MISSING}}", {})


def test_llm_function_name_stable():
    name = llm_function_name("a1b2c3d4-e5f6-7890-abcd-ef1234567890", "Check Calendar!")
    assert name.startswith("tool_a1b2c3d4_")
    assert len(name) <= 64


def test_build_parameters_from_query_params():
    cfg = parse_tool_config(
        {
            "schema_version": 1,
            "url": "https://api.example.com/search",
            "query_parameters": [
                {"name": "q", "description": "query", "type": "string", "required": True},
            ],
        }
    )
    schema = build_parameters_schema(cfg)
    assert "q" in schema["properties"]
    assert schema["required"] == ["q"]


def test_collect_tool_ids_from_spec():
    spec = ConversationSpecV1(
        mode="graph",
        tool_ids=["t1"],
        nodes=[
            SpecNode(id="start", system_prompt="hi", tool_ids=["t2"]),
        ],
        edges=[],
        entry_node_id="start",
    )
    assert set(collect_tool_ids_from_spec(spec)) == {"t1", "t2"}
    assert tool_ids_for_node(spec, "start") == ["t2"]


def test_build_http_tool_definitions_filters_by_node():
    rows = [
        {
            "id": "11111111-1111-1111-1111-111111111111",
            "name": "Weather",
            "description": "Get weather",
            "kind": "http",
            "config": {
                "schema_version": 1,
                "url": "https://api.example.com/weather",
                "method": "GET",
            },
        }
    ]
    tools, by_fn = build_http_tool_definitions(
        rows,
        node_tool_ids=["11111111-1111-1111-1111-111111111111"],
    )
    assert len(tools) == 1
    assert tools[0].name in by_fn


def test_ssrf_blocks_localhost():
    with pytest.raises(ValueError, match="not allowed"):
        validate_url_ssrf("http://localhost/hook")
