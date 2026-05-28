from fastapi.testclient import TestClient

from ai_sdr_agent.app import create_app
from ai_sdr_agent.config import SDRSettings


def build_test_settings() -> SDRSettings:
    return SDRSettings(llm_provider="stub")


def test_healthz_reports_ok():
    client = TestClient(create_app(build_test_settings()))
    response = client.get("/healthz")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["llm_provider"] == "stub"


def test_latency_analytics_endpoint():
    client = TestClient(create_app(build_test_settings()))
    r = client.get("/analytics/latency?recent_limit=10")
    assert r.status_code == 200
    data = r.json()
    assert "latency_graph_ms" in data
    assert "web_voice" in data
