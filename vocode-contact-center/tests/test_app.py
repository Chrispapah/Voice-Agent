import asyncio

from fastapi.testclient import TestClient
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.synthesizer import ElevenLabsSynthesizerConfig

from vocode_contact_center.agent import ContactCenterAgent, ContactCenterAgentConfig
from vocode_contact_center.app import build_inbound_call_config
from vocode_contact_center.app import create_app
from vocode_contact_center.settings import ContactCenterSettings


def test_app_starts_in_degraded_mode_without_provider_keys():
    app = create_app(
        ContactCenterSettings(
            base_url=None,
            ngrok_auth_token=None,
            twilio_account_sid=None,
            twilio_auth_token=None,
            deepgram_api_key=None,
            elevenlabs_api_key=None,
            elevenlabs_voice_id=None,
            langchain_provider="openai",
            openai_api_key=None,
        )
    )

    client = TestClient(app)
    response = client.get("/healthz")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "degraded"
    assert "TWILIO_ACCOUNT_SID" in payload["missing_runtime_values"]
    assert payload["stt_provider"] == "deepgram"
    assert payload["inbound_call_url"] is None


def test_inbound_call_config_uses_elevenlabs_telephone_synthesizer():
    settings = ContactCenterSettings(
        base_url="demo.example.com",
        twilio_account_sid="sid",
        twilio_auth_token="token",
        deepgram_api_key="deepgram",
        elevenlabs_api_key="eleven",
        elevenlabs_voice_id="voice",
        langchain_provider="openai",
        openai_api_key="openai",
    )

    config = build_inbound_call_config(settings)

    assert isinstance(config.synthesizer_config, ElevenLabsSynthesizerConfig)
    assert config.synthesizer_config.voice_id == "voice"


def test_healthz_includes_sip_domain_and_full_inbound_url():
    app = create_app(
        ContactCenterSettings(
            base_url="demo.example.com",
            twilio_account_sid="sid",
            twilio_auth_token="token",
            twilio_sip_domain="voiceagentpapas.sip.twilio.com",
            deepgram_api_key="deepgram",
            elevenlabs_api_key="eleven",
            elevenlabs_voice_id="voice",
            langchain_provider="openai",
            openai_api_key="openai",
        )
    )

    client = TestClient(app)
    payload = client.get("/healthz").json()

    assert payload["status"] == "ok"
    assert payload["twilio_sip_domain"] == "voiceagentpapas.sip.twilio.com"
    assert payload["inbound_call_url"] == "https://demo.example.com/inbound_call"


def test_contact_center_agent_returns_fallback_for_handoff_request():
    agent = ContactCenterAgent(
        ContactCenterAgentConfig(
            initial_message=BaseMessage(text="Hello"),
            prompt_preamble="You are helpful.",
            model_name="gpt-4o-mini",
            provider="openai",
        )
    )

    message, should_stop = asyncio.run(
        agent.respond("I want a human agent", conversation_id="conversation-1")
    )

    assert "human agent" in message.lower()
    assert should_stop is False
