from __future__ import annotations

import os
from urllib.parse import urlparse

from pydantic_settings import BaseSettings, SettingsConfigDict

from vocode_contact_center.prompts import DEFAULT_AGENT_PROMPT


class ContactCenterSettings(BaseSettings):
    port: int = 3000
    base_url: str | None = None
    ngrok_auth_token: str | None = None
    railway_public_domain: str | None = None

    twilio_account_sid: str | None = None
    twilio_auth_token: str | None = None
    twilio_sip_domain: str | None = None
    sms_adapter_mode: str = "stub"
    twilio_sms_from_number: str | None = None
    twilio_messaging_service_sid: str | None = None
    sms_default_region: str | None = None
    registration_confirmation_sms_template: str = (
        "Hi {full_name}, your registration request has been confirmed. "
        "We'll follow up with the next steps shortly."
    )
    redis_url: str | None = None

    deepgram_api_key: str | None = None
    deepgram_model: str = "nova-2"
    deepgram_vad_threshold_ms: int = 140
    deepgram_utterance_cutoff_ms: int = 600
    deepgram_time_cutoff_seconds: float = 0.12
    deepgram_post_punctuation_time_seconds: float = 0.05
    deepgram_single_utterance_for_first_response: bool = True

    elevenlabs_api_key: str | None = None
    elevenlabs_voice_id: str | None = None
    elevenlabs_model_id: str = "eleven_turbo_v2"
    elevenlabs_use_websocket: bool = True
    elevenlabs_optimize_streaming_latency: int = 4

    conversation_orchestrator: str = "graph"
    langchain_provider: str = "groq"
    langchain_model_name: str = "llama-3.3-70b-versatile"
    langchain_temperature: float = 0.2
    langchain_max_tokens: int = 64
    langchain_recent_message_limit: int = 6
    langchain_summary_max_messages: int = 12
    langchain_summary_max_chars: int = 600
    require_streaming_synthesizer: bool = True
    non_streaming_chunk_min_words: int = 3
    non_streaming_chunk_max_words: int = 8
    non_streaming_chunk_min_chars: int = 12
    realtime_enabled: bool = True
    realtime_transport: str = "websocket"
    realtime_input_mode: str = "partial_transcripts"
    realtime_audio_encoding: str = "linear16"
    realtime_sample_rate: int = 16000
    realtime_allow_partial_responses: bool = True
    realtime_partial_response_min_words: int = 3
    realtime_partial_response_min_chars: int = 12

    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    google_api_key: str | None = None
    groq_api_key: str | None = None

    agent_name: str = "AI Contact Center"
    agent_initial_message: str = (
        "Thanks for calling. I can help with general information, account support like registration "
        "or login, the latest announcements, or feedback and contact options. What would you like help with today?"
    )
    agent_prompt_preamble: str = DEFAULT_AGENT_PROMPT
    transfer_phone_number: str | None = None
    information_store_website_url: str = "https://example.com/store-locations"
    information_products_pdf_url: str = "https://example.com/products.pdf"
    announcements_message: str = (
        "I can share the latest announcements with you."
    )
    feedback_question_prompt: str = (
        "Before we finish, would you like to return to the chat, or would you prefer contact options for more help?"
    )
    voicebot_adapter_mode: str = "stub"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def missing_runtime_values(self) -> list[str]:
        missing = []

        required = {
            "TWILIO_ACCOUNT_SID": self.twilio_account_sid,
            "TWILIO_AUTH_TOKEN": self.twilio_auth_token,
            "DEEPGRAM_API_KEY": self.deepgram_api_key,
            "ELEVENLABS_API_KEY": self.elevenlabs_api_key,
            "ELEVENLABS_VOICE_ID": self.elevenlabs_voice_id,
        }

        for key, value in required.items():
            if not value:
                missing.append(key)

        provider_requirements = {
            "openai": ("OPENAI_API_KEY", self.openai_api_key),
            "anthropic": ("ANTHROPIC_API_KEY", self.anthropic_api_key),
            "google_genai": ("GOOGLE_API_KEY", self.google_api_key),
            "groq": ("GROQ_API_KEY", self.groq_api_key),
        }
        provider_key = self.langchain_provider.strip().lower()
        provider_requirement = provider_requirements.get(provider_key)
        if provider_requirement is not None:
            name, value = provider_requirement
            if not value:
                missing.append(name)

        if not self.base_url and not self.ngrok_auth_token and not self.railway_public_domain:
            missing.append("BASE_URL or NGROK_AUTH_TOKEN")

        return missing

    def missing_realtime_values(self) -> list[str]:
        missing = []

        required = {
            "ELEVENLABS_API_KEY": self.elevenlabs_api_key,
            "ELEVENLABS_VOICE_ID": self.elevenlabs_voice_id,
        }

        for key, value in required.items():
            if not value:
                missing.append(key)

        provider_requirements = {
            "openai": ("OPENAI_API_KEY", self.openai_api_key),
            "anthropic": ("ANTHROPIC_API_KEY", self.anthropic_api_key),
            "google_genai": ("GOOGLE_API_KEY", self.google_api_key),
            "groq": ("GROQ_API_KEY", self.groq_api_key),
        }
        provider_key = self.langchain_provider.strip().lower()
        provider_requirement = provider_requirements.get(provider_key)
        if provider_requirement is not None:
            name, value = provider_requirement
            if not value:
                missing.append(name)

        return missing

    def missing_sms_values(self) -> list[str]:
        if self.sms_adapter_mode.strip().lower() != "twilio":
            return []

        missing = []
        required = {
            "TWILIO_ACCOUNT_SID": self.twilio_account_sid,
            "TWILIO_AUTH_TOKEN": self.twilio_auth_token,
        }

        for key, value in required.items():
            if not value:
                missing.append(key)

        if not self.twilio_sms_from_number and not self.twilio_messaging_service_sid:
            missing.append("TWILIO_SMS_FROM_NUMBER or TWILIO_MESSAGING_SERVICE_SID")

        return missing

    def normalized_base_url(self) -> str | None:
        candidate = self.base_url or self.railway_public_domain
        if not candidate:
            return None
        return candidate.removeprefix("https://").removeprefix("http://").rstrip("/")

    def inbound_call_url(self, public_base_url: str | None = None) -> str | None:
        base_url = public_base_url or self.normalized_base_url()
        if not base_url:
            return None
        return f"https://{base_url}/inbound_call"

    def apply_redis_env(self) -> None:
        if not self.redis_url:
            return

        parsed = urlparse(self.redis_url)
        if parsed.hostname:
            os.environ["REDISHOST"] = parsed.hostname
        if parsed.port:
            os.environ["REDISPORT"] = str(parsed.port)
        if parsed.username:
            os.environ["REDISUSER"] = parsed.username
        if parsed.password:
            os.environ["REDISPASSWORD"] = parsed.password
        # Vocode currently casts REDISSSL with bool(os.environ.get(...)), so any non-empty
        # string becomes truthy. Use an empty string to disable SSL cleanly for redis:// URLs.
        os.environ["REDISSSL"] = "1" if parsed.scheme == "rediss" else ""
