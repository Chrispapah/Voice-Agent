from fastapi.testclient import TestClient

from ai_sdr_agent.app import create_app
from ai_sdr_agent.config import SDRSettings


def build_test_settings() -> SDRSettings:
    return SDRSettings(
        llm_provider="stub",
        use_stub_integrations=True,
        use_redis_config_manager=False,
        base_url=None,
        twilio_account_sid=None,
        twilio_auth_token=None,
        twilio_phone_number=None,
        deepgram_api_key=None,
        elevenlabs_api_key=None,
        elevenlabs_voice_id=None,
    )


def test_healthz_reports_degraded_without_telephony_credentials():
    client = TestClient(create_app(build_test_settings()))
    response = client.get("/healthz")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "degraded"
    assert "TWILIO_ACCOUNT_SID" in payload["missing_runtime_values"]


def test_start_session_and_run_turns():
    client = TestClient(create_app(build_test_settings()))

    response = client.post("/sessions", json={"lead_id": "lead-001"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["state"]["current_node"] == "greeting"
    conversation_id = payload["conversation_id"]

    response = client.post(
        f"/sessions/{conversation_id}/turns",
        json={"human_input": "Yes, I run sales operations."},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["state"]["current_node"] == "qualify_lead"


def test_unknown_session_returns_404():
    client = TestClient(create_app(build_test_settings()))
    response = client.get("/sessions/missing-id")

    assert response.status_code == 404
