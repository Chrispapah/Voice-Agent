from __future__ import annotations

import os

from dotenv import load_dotenv
from fastapi import FastAPI
from loguru import logger
from pyngrok import ngrok
from vocode.logging import configure_pretty_logging
from vocode.streaming.models.agent import AgentConfig
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.synthesizer import ElevenLabsSynthesizerConfig
from vocode.streaming.models.telephony import TwilioConfig
from vocode.streaming.models.transcriber import (
    DeepgramTranscriberConfig,
)
from vocode.streaming.transcriber.deepgram_transcriber import (
    DeepgramEndpointingConfig,
    TimeSilentConfig,
)
from vocode.streaming.telephony.config_manager.redis_config_manager import RedisConfigManager
from vocode.streaming.telephony.server.base import TwilioInboundCallConfig

from vocode_contact_center.agent import ContactCenterAgentConfig
from vocode_contact_center.agent_factory import ContactCenterAgentFactory
from vocode_contact_center.latency_tracker import conversation_latency_tracker
from vocode_contact_center.orchestration import (
    ConversationOrchestrator,
    HybridConversationOrchestratorService,
)
from vocode_contact_center.realtime_worker import (
    RealtimeSessionManager,
    RealtimeSnapshot,
    create_realtime_router,
)
from vocode_contact_center.settings import ContactCenterSettings
from vocode_contact_center.telephony_server import LatencyTrackingTelephonyServer
from vocode_contact_center.voicebot_graph.adapters.stub import StubSmsSender
from vocode_contact_center.voicebot_graph.adapters.twilio_sms import TwilioSmsSender
load_dotenv()
configure_pretty_logging()


class DisabledRealtimeSessionManager:
    def __init__(self) -> None:
        self.legacy_telephony_available = False

    def snapshot(self) -> RealtimeSnapshot:
        return RealtimeSnapshot(
            total_sessions_created=0,
            active_sessions=0,
            total_interruptions=0,
            completed_responses=0,
        )


def ensure_nltk_resources(settings: ContactCenterSettings) -> list[str]:
    # Vocode's ElevenLabs synthesizer uses NLTK tokenization for message cutoff logic.
    # Avoid downloading these on Railway request-serving startup, which can cause cold-start
    # timeouts for inbound Twilio webhooks. Pre-bake them in the image or opt in explicitly.
    import nltk
    from nltk.data import find

    required_resources = (
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab/english", "punkt_tab"),
    )

    missing_resources: list[str] = []
    for resource_path, download_name in required_resources:
        try:
            find(resource_path)
        except LookupError:
            missing_resources.append(download_name)

    if not missing_resources:
        return []

    if not settings.should_auto_download_nltk():
        logger.warning(
            "NLTK resources missing at startup and auto-download is disabled: {}. "
            "Voice webhook cold starts stay fast, but the synthesizer may need these resources preinstalled.",
            ", ".join(missing_resources),
        )
        return missing_resources

    unresolved_resources: list[str] = []
    for download_name in missing_resources:
        logger.info("Downloading NLTK resource: {}", download_name)
        download_succeeded = bool(nltk.download(download_name, quiet=True))
        try:
            if download_name == "punkt":
                find("tokenizers/punkt")
            else:
                find("tokenizers/punkt_tab/english")
        except LookupError:
            download_succeeded = False
        if not download_succeeded:
            unresolved_resources.append(download_name)

    if unresolved_resources:
        logger.warning(
            "NLTK resources are still unavailable after startup download attempt: {}",
            ", ".join(unresolved_resources),
        )
    return unresolved_resources


def apply_runtime_env(settings: ContactCenterSettings) -> None:
    if settings.openai_api_key:
        os.environ["OPENAI_API_KEY"] = settings.openai_api_key
    if settings.anthropic_api_key:
        os.environ["ANTHROPIC_API_KEY"] = settings.anthropic_api_key
    if settings.google_api_key:
        os.environ["GOOGLE_API_KEY"] = settings.google_api_key
    if settings.groq_api_key:
        os.environ["GROQ_API_KEY"] = settings.groq_api_key
    settings.apply_redis_env()


def resolve_base_url(settings: ContactCenterSettings) -> str | None:
    base_url = settings.normalized_base_url()
    if base_url:
        return base_url

    if not settings.ngrok_auth_token:
        return None

    ngrok.set_auth_token(settings.ngrok_auth_token)
    tunnel = ngrok.connect(settings.port)
    public_url = tunnel.public_url.removeprefix("https://").removeprefix("http://")
    logger.info("ngrok tunnel {} -> http://127.0.0.1:{}", public_url, settings.port)
    return public_url


def build_agent_config(settings: ContactCenterSettings) -> AgentConfig:
    return ContactCenterAgentConfig(
        initial_message=BaseMessage(text=settings.agent_initial_message),
        prompt_preamble=settings.agent_prompt_preamble,
        model_name=settings.langchain_model_name,
        provider=settings.langchain_provider,
        temperature=settings.langchain_temperature,
        max_tokens=settings.langchain_max_tokens,
        recent_message_limit=settings.langchain_recent_message_limit,
        summary_max_messages=settings.langchain_summary_max_messages,
        summary_max_chars=settings.langchain_summary_max_chars,
        non_streaming_chunk_min_words=settings.non_streaming_chunk_min_words,
        non_streaming_chunk_max_words=settings.non_streaming_chunk_max_words,
        non_streaming_chunk_min_chars=settings.non_streaming_chunk_min_chars,
        require_streaming_synthesizer=settings.require_streaming_synthesizer,
        transfer_phone_number=settings.transfer_phone_number,
        generate_responses=True,
        end_conversation_on_goodbye=True,
        interrupt_sensitivity="low",
        allow_agent_to_be_cut_off=settings.allow_agent_to_be_cut_off,
    )


def build_inbound_call_config(settings: ContactCenterSettings) -> TwilioInboundCallConfig:
    agent_config = build_agent_config(settings)
    return TwilioInboundCallConfig(
        url="/inbound_call",
        agent_config=agent_config,
        transcriber_config=DeepgramTranscriberConfig.from_telephone_input_device(
            endpointing_config=DeepgramEndpointingConfig(
                vad_threshold_ms=settings.deepgram_vad_threshold_ms,
                utterance_cutoff_ms=settings.deepgram_utterance_cutoff_ms,
                time_silent_config=TimeSilentConfig(
                    time_cutoff_seconds=settings.deepgram_time_cutoff_seconds,
                    post_punctuation_time_seconds=settings.deepgram_post_punctuation_time_seconds,
                ),
                use_single_utterance_endpointing_for_first_utterance=(
                    settings.deepgram_single_utterance_for_first_response
                ),
            ),
            api_key=settings.deepgram_api_key,
            model=settings.deepgram_model,
            mute_during_speech=settings.deepgram_mute_during_speech,
        ),
        synthesizer_config=ElevenLabsSynthesizerConfig.from_telephone_output_device(
            api_key=settings.elevenlabs_api_key,
            voice_id=settings.elevenlabs_voice_id,
            model_id=settings.elevenlabs_model_id,
            optimize_streaming_latency=settings.elevenlabs_optimize_streaming_latency,
            experimental_websocket=settings.elevenlabs_use_websocket,
        ),
        twilio_config=TwilioConfig(
            account_sid=settings.twilio_account_sid,
            auth_token=settings.twilio_auth_token,
        ),
    )


def build_conversation_orchestrator(
    settings: ContactCenterSettings,
) -> ConversationOrchestrator:
    sms_sender = build_sms_sender(settings)
    return HybridConversationOrchestratorService(settings, sms_sender=sms_sender)


def build_sms_sender(settings: ContactCenterSettings):
    if settings.sms_adapter_mode.strip().lower() != "twilio":
        return StubSmsSender()

    missing = settings.missing_sms_values()
    if missing:
        logger.warning(
            "SMS sender is configured for Twilio but missing values: {}. Falling back to stub sender.",
            ", ".join(missing),
        )
        return StubSmsSender()

    return TwilioSmsSender(settings)


def create_app(settings: ContactCenterSettings | None = None) -> FastAPI:
    settings = settings or ContactCenterSettings()
    apply_runtime_env(settings)
    missing_nltk_resources = ensure_nltk_resources(settings)
    logger.info(
        "create_app initialized base_url_candidate={} railway_public_domain={} realtime_enabled={} sms_adapter_mode={} twilio_message_channel={} missing_nltk_resources={}",
        settings.normalized_base_url(),
        settings.railway_public_domain,
        settings.realtime_enabled,
        settings.sms_adapter_mode,
        settings.normalized_twilio_message_channel(),
        missing_nltk_resources,
    )

    app = FastAPI(title="Vocode AI Contact Center", version="0.1.0")
    conversation_orchestrator = build_conversation_orchestrator(settings)
    app.state.voicebot_service = conversation_orchestrator
    app.state.conversation_orchestrator = conversation_orchestrator
    realtime_missing = settings.missing_realtime_values() if settings.realtime_enabled else ["REALTIME_DISABLED"]
    realtime_ready = settings.realtime_enabled and not realtime_missing
    realtime_manager: RealtimeSessionManager | DisabledRealtimeSessionManager
    if realtime_ready:
        try:
            realtime_manager = RealtimeSessionManager(
                settings,
                conversation_orchestrator=conversation_orchestrator,
                legacy_telephony_available=False,
            )
        except Exception as exc:
            logger.exception("Realtime voice disabled due to initialization failure: {}", exc)
            realtime_missing = ["REALTIME_INITIALIZATION_FAILED"]
            realtime_ready = False
            realtime_manager = DisabledRealtimeSessionManager()
    else:
        realtime_manager = DisabledRealtimeSessionManager()
    app.state.realtime_manager = realtime_manager
    app.state.realtime_ready = realtime_ready
    app.state.missing_realtime_values = [] if realtime_ready else realtime_missing
    app.state.missing_nltk_resources = missing_nltk_resources
    app.include_router(
        create_realtime_router(
            settings,
            manager=realtime_manager,
            realtime_ready=realtime_ready,
        )
    )

    @app.get("/healthz")
    async def healthz():
        missing = settings.missing_runtime_values()
        return {
            "status": "ok" if not missing else "degraded",
            "public_base_url": app.state.public_base_url,
            "inbound_call_url": settings.inbound_call_url(app.state.public_base_url),
            "missing_runtime_values": missing,
            "twilio_sip_domain": settings.twilio_sip_domain,
            "twilio_webhook_path": "/inbound_call",
            "twilio_sip_http_method": "POST",
            "stt_provider": "deepgram",
            "require_streaming_synthesizer": settings.require_streaming_synthesizer,
            "realtime_enabled": settings.realtime_enabled,
            "realtime_ready": app.state.realtime_ready,
            "realtime_transport": settings.realtime_transport,
            "realtime_input_mode": settings.realtime_input_mode,
            "missing_realtime_values": app.state.missing_realtime_values,
            "stt_note": "Vocode docs support ElevenLabs for TTS, not STT/transcriber.",
            "conversation_orchestrator": "hybrid",
            "sms_adapter_mode": settings.sms_adapter_mode,
            "twilio_message_channel": settings.normalized_twilio_message_channel(),
            "missing_sms_values": settings.missing_sms_values(),
            "nltk_auto_download": settings.should_auto_download_nltk(),
            "missing_nltk_resources": app.state.missing_nltk_resources,
        }

    @app.get("/latencyz")
    async def latencyz():
        snapshot = conversation_latency_tracker.snapshot()
        snapshot["realtime"] = app.state.realtime_manager.snapshot().model_dump()
        return snapshot

    missing = settings.missing_runtime_values()
    base_url = resolve_base_url(settings)
    if base_url is None and "BASE_URL or NGROK_AUTH_TOKEN" not in missing:
        missing.append("BASE_URL or NGROK_AUTH_TOKEN")

    app.state.missing_runtime_values = missing
    app.state.telephony_enabled = not missing
    app.state.public_base_url = base_url
    logger.info(
        "App readiness preflight public_base_url={} telephony_enabled={} missing_runtime_values={} missing_realtime_values={} missing_sms_values={}",
        base_url,
        app.state.telephony_enabled,
        missing,
        app.state.missing_realtime_values,
        settings.missing_sms_values(),
    )

    @app.on_event("startup")
    async def log_startup_readiness():
        logger.info(
            "Application startup complete telephony_enabled={} public_base_url={} inbound_call_url={} realtime_ready={} missing_runtime_values={} missing_nltk_resources={}",
            app.state.telephony_enabled,
            app.state.public_base_url,
            settings.inbound_call_url(app.state.public_base_url),
            app.state.realtime_ready,
            app.state.missing_runtime_values,
            app.state.missing_nltk_resources,
        )

    @app.on_event("shutdown")
    async def log_shutdown_readiness():
        logger.info(
            "Application shutdown telephony_enabled={} public_base_url={} realtime_ready={}",
            app.state.telephony_enabled,
            app.state.public_base_url,
            app.state.realtime_ready,
        )

    if missing or not base_url:
        logger.warning(
            "Telephony routes not mounted. Missing runtime config: {}",
            ", ".join(missing) if missing else "base URL",
        )
        return app

    config_manager = RedisConfigManager()
    inbound_call_config = build_inbound_call_config(settings)
    app.state.realtime_manager.legacy_telephony_available = True
    telephony_server = LatencyTrackingTelephonyServer(
        base_url=base_url,
        config_manager=config_manager,
        inbound_call_configs=[inbound_call_config],
        agent_factory=ContactCenterAgentFactory(
            conversation_orchestrator=conversation_orchestrator
        ),
    )
    app.include_router(telephony_server.get_router())
    logger.info(
        "Telephony routes mounted inbound_call_url={} events_url=https://{}/events",
        settings.inbound_call_url(app.state.public_base_url),
        base_url,
    )

    @app.get("/")
    async def root():
        return {
            "status": "ready",
            "message": "Twilio can post inbound SIP call webhooks to /inbound_call.",
            "inbound_call_url": settings.inbound_call_url(app.state.public_base_url),
            "twilio_sip_domain": settings.twilio_sip_domain,
            "realtime_session_endpoint": "/realtime/sessions",
        }

    return app
