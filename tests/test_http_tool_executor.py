from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_sdr_agent.services.http_tool_executor import execute_http_tool
from ai_sdr_agent.services.tool_config import HttpToolConfigV1


@pytest.mark.asyncio
async def test_execute_http_tool_success():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = '{"temperature": 72}'
    mock_response.headers = {"content-type": "application/json"}
    mock_response.json.return_value = {"temperature": 72}

    mock_client = AsyncMock()
    mock_client.request = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    config = HttpToolConfigV1(
        schema_version=1,
        url="https://api.example.com/weather",
        method="GET",
    )
    with patch("ai_sdr_agent.services.http_tool_executor.httpx.AsyncClient", return_value=mock_client):
        result = await execute_http_tool(config=config, args={}, env={})
    assert "72" in result
