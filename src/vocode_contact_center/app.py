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
    PunctuationEndpointingConfig,
)
from vocode.streaming.telephony.config_manager.redis_config_manager import RedisConfigManager
from vocode.streaming.telephony.server.base import TelephonyServer, TwilioInboundCallConfig

from vocode_contact_center.agent import ContactCenterAgentConfig
from vocode_contact_center.agent_factory import ContactCenterAgentFactory
from vocode_contact_center.settings import ContactCenterSettings

load_dotenv()
configure_pretty_logging()


def ensure_nltk_resources() -> None:
    # Vocode's ElevenLabs synthesizer uses NLTK tokenization for message cutoff logic.
    # Railway images start empty, so make sure the required tokenizers exist before calls.
    import nltk
    from nltk.data import find

    required_resources = (
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab/english", "punkt_tab"),
    )

    for resource_path, download_name in required_resources:
        try:
            find(resource_path)
        except LookupError:
            logger.info("Downloading NLTK resource: {}", download_name)
            nltk.download(download_name, quiet=True)


def apply_runtime_env(settings: ContactCenterSettings) -> None:
    if settings.openai_api_key:
        os.environ["OPENAI_API_KEY"] = settings.openai_api_key
    if settings.anthropic_api_key:
        os.environ["ANTHROPIC_API_KEY"] = settings.anthropic_api_key
    if settings.google_api_key:
        os.environ["GOOGLE_API_KEY"] = settings.google_api_key
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
        transfer_phone_number=settings.transfer_phone_number,
        generate_responses=True,
        end_conversation_on_goodbye=True,
        interrupt_sensitivity="high",
    )


def build_inbound_call_config(settings: ContactCenterSettings) -> TwilioInboundCallConfig:
    return TwilioInboundCallConfig(
        url="/inbound_call",
        agent_config=build_agent_config(settings),
        transcriber_config=DeepgramTranscriberConfig.from_telephone_input_device(
            endpointing_config=PunctuationEndpointingConfig(),
            api_key=settings.deepgram_api_key,
            model=settings.deepgram_model,
        ),
        synthesizer_config=ElevenLabsSynthesizerConfig.from_telephone_output_device(
            api_key=settings.elevenlabs_api_key,
            voice_id=settings.elevenlabs_voice_id,
            model_id=settings.elevenlabs_model_id,
            optimize_streaming_latency=3,
            experimental_websocket=settings.elevenlabs_use_websocket,
        ),
        twilio_config=TwilioConfig(
            account_sid=settings.twilio_account_sid,
            auth_token=settings.twilio_auth_token,
        ),
    )


def create_app(settings: ContactCenterSettings | None = None) -> FastAPI:
    settings = settings or ContactCenterSettings()
    apply_runtime_env(settings)
    ensure_nltk_resources()

    app = FastAPI(title="Vocode AI Contact Center", version="0.1.0")

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
            "stt_note": "Vocode docs support ElevenLabs for TTS, not STT/transcriber.",
        }

    missing = settings.missing_runtime_values()
    base_url = resolve_base_url(settings)
    if base_url is None and "BASE_URL or NGROK_AUTH_TOKEN" not in missing:
        missing.append("BASE_URL or NGROK_AUTH_TOKEN")

    app.state.missing_runtime_values = missing
    app.state.telephony_enabled = not missing
    app.state.public_base_url = base_url

    if missing or not base_url:
        logger.warning(
            "Telephony routes not mounted. Missing runtime config: {}",
            ", ".join(missing) if missing else "base URL",
        )
        return app

    config_manager = RedisConfigManager()
    telephony_server = TelephonyServer(
        base_url=base_url,
        config_manager=config_manager,
        inbound_call_configs=[build_inbound_call_config(settings)],
        agent_factory=ContactCenterAgentFactory(),
    )
    app.include_router(telephony_server.get_router())

    @app.get("/")
    async def root():
        return {
            "status": "ready",
            "message": "Twilio can post inbound SIP call webhooks to /inbound_call.",
            "inbound_call_url": settings.inbound_call_url(app.state.public_base_url),
            "twilio_sip_domain": settings.twilio_sip_domain,
        }

    return app
