import asyncio

from fastapi.testclient import TestClient
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.synthesizer import ElevenLabsSynthesizerConfig

from vocode_contact_center.agent import (
    ContactCenterAgent,
    ContactCenterAgentConfig,
    build_call_context,
)
from vocode_contact_center.app import (
    build_conversation_orchestrator,
    build_inbound_call_config,
    create_app,
)
from vocode_contact_center.latency_tracker import ConversationLatencyTracker
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
    assert payload["realtime_enabled"] is True
    assert payload["realtime_ready"] is False


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


def test_contact_center_agent_uses_preloaded_call_context():
    agent = ContactCenterAgent(
        ContactCenterAgentConfig(
            initial_message=BaseMessage(text="Hello"),
            prompt_preamble="You are helpful.",
            model_name="gpt-4o-mini",
            provider="openai",
            call_context=build_call_context(
                from_phone="+1234567890",
                to_phone="+1098765432",
            ),
        )
    )

    assert "Caller number: +1234567890" in agent._get_call_context()
    assert "Dialed number: +1098765432" in agent._get_call_context()


def test_healthz_reports_streaming_requirement_and_latency_endpoint_is_available():
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
            require_streaming_synthesizer=True,
        )
    )

    client = TestClient(app)
    health_payload = client.get("/healthz").json()
    latency_payload = client.get("/latencyz").json()

    assert health_payload["require_streaming_synthesizer"] is True
    assert health_payload["realtime_transport"] == "websocket"
    assert "segments" in latency_payload
    assert "realtime" in latency_payload
    assert latency_payload["active_conversations"] == 0


def test_build_conversation_orchestrator_uses_hybrid_service(monkeypatch):
    captured = {}

    class FakeHybridOrchestrator:
        def __init__(self, settings, *, sms_sender=None):
            captured["settings"] = settings
            captured["sms_sender"] = sms_sender

    monkeypatch.setattr(
        "vocode_contact_center.app.HybridConversationOrchestratorService",
        FakeHybridOrchestrator,
    )

    orchestrator = build_conversation_orchestrator(
        ContactCenterSettings(
            langchain_provider="openai",
            openai_api_key="openai",
        )
    )

    assert isinstance(orchestrator, FakeHybridOrchestrator)
    assert captured["sms_sender"] is not None


def test_app_initializes_shared_voicebot_service():
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
            openai_api_key="openai",
        )
    )

    assert hasattr(app.state, "voicebot_service")
    assert hasattr(app.state, "conversation_orchestrator")


def test_latency_tracker_snapshot_aggregates_segment_timings():
    tracker = ConversationLatencyTracker()

    tracker.mark_call_started(
        "conversation-1",
        from_phone="+1234567890",
        to_phone="+1098765432",
    )
    tracker.mark_audio_received("conversation-1")
    tracker.mark_transcription_final("conversation-1", "hello there")
    tracker.mark_first_model_token("conversation-1")
    tracker.mark_first_llm_token(
        "conversation-1",
        using_input_streaming_synthesizer=True,
    )
    tracker.mark_first_tts_chunk("conversation-1")

    snapshot = tracker.snapshot()

    assert snapshot["segments"]["audio_to_final_ms"]["count"] == 1
    assert snapshot["segments"]["final_to_first_model_token_ms"]["count"] == 1
    assert snapshot["segments"]["final_to_first_llm_ms"]["count"] == 1
    assert snapshot["segments"]["first_llm_to_tts_ms"]["count"] == 1
    assert snapshot["segments"]["final_to_first_tts_ms"]["count"] == 1
    assert snapshot["active_conversation_details"]["conversation-1"]["from_phone"] == "+1234567890"
    assert snapshot["active_conversation_details"]["conversation-1"]["to_phone"] == "+1098765432"
