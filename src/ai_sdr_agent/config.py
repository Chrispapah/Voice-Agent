from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class SDRSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "AI SDR Voice Agent"
    app_env: str = "development"
    host: str = "0.0.0.0"
    port: int = 3000
    base_url: str | None = None

    llm_provider: Literal["openai", "anthropic", "groq", "stub"] = "stub"
    llm_model_name: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.4
    llm_max_tokens: int = 220

    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    groq_api_key: str | None = None

    twilio_account_sid: str | None = None
    twilio_auth_token: str | None = None
    twilio_phone_number: str | None = None
    twilio_record_calls: bool = False

    stt_provider: Literal["deepgram", "google"] = "google"

    google_speech_api_key: str | None = None
    google_speech_language_code: str = "en-US"
    # phone_call + enhanced matches Twilio mulaw telephony (see google_speech_transcriber).
    google_speech_model: str | None = "phone_call"
    # Client-side: silence after last interim before a "final" (lower = snappier, more mid-sentence cuts). Floored in code at 40ms.
    google_utterance_silence_ms: int = 300
    google_mute_during_speech: bool = True

    deepgram_api_key: str | None = None
    deepgram_language: str = "en-US"
    deepgram_model: str = "nova-2"
    # Endpointing: lower = final transcript sooner after a pause (more false end-of-turn / mid-sentence cuts).
    # Deepgram URL "endpointing" = vad_threshold_ms; time/punctuation = Vocode fallbacks when speech_final lags.
    # utterance_cutoff_ms → Deepgram utterance_end_ms (Vocode enforces ≥1000). Higher = more tolerance for
    # pauses within a thought; 2000ms has been a better default than 900ms on live calls.
    deepgram_vad_threshold_ms: int = 80
    deepgram_utterance_cutoff_ms: int = 2000
    deepgram_time_cutoff_seconds: float = 0.08
    deepgram_post_punctuation_time_seconds: float = 0.035
    deepgram_single_utterance_for_first_response: bool = True
    deepgram_mute_during_speech: bool = True

    tts_provider: Literal["elevenlabs", "azure"] = "elevenlabs"
    elevenlabs_api_key: str | None = None
    elevenlabs_voice_id: str | None = None
    elevenlabs_model_id: str = "eleven_turbo_v2"
    elevenlabs_use_websocket: bool = False
    elevenlabs_optimize_streaming_latency: int = 4

    azure_speech_key: str | None = None
    azure_speech_region: str | None = None
    azure_voice_name: str = "en-US-JennyNeural"

    redis_url: str | None = None
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/ai_sdr"
    supabase_url: str | None = None
    supabase_jwt_secret: str = "CHANGE-ME-set-SUPABASE_JWT_SECRET-in-env"

    google_calendar_credentials_json: str | None = None
    sendgrid_api_key: str | None = None
    smtp_host: str | None = None
    smtp_port: int = 587
    smtp_username: str | None = None
    smtp_password: str | None = None
    crm_api_key: str | None = None

    default_calendar_id: str = "sales-team"
    default_sales_rep_name: str = "Sales Team"
    default_sender_email: str = "sales@example.com"
    default_from_name: str = "AI SDR"
    outbound_caller_name: str = "John"
    initial_greeting: str = (
        "Hi, this is John — I know I'm calling out of the blue. "
        "Do you have 30 seconds so I can tell you why I'm reaching out?"
    )
    max_objection_attempts: int = 2
    max_call_turns: int = 12

    use_redis_config_manager: bool = True
    use_stub_integrations: bool = True
    auto_send_follow_up_email: bool = True
    auto_update_crm: bool = True
    prefetch_calendar_days: int = 5

    def normalized_base_url(self) -> str | None:
        if not self.base_url:
            return None
        value = self.base_url.strip()
        if value.startswith("http://"):
            value = value[len("http://") :]
        if value.startswith("https://"):
            value = value[len("https://") :]
        return value.rstrip("/")

    def missing_runtime_values(self) -> list[str]:
        missing: list[str] = []
        if self.llm_provider == "openai" and not self.openai_api_key:
            missing.append("OPENAI_API_KEY")
        if self.llm_provider == "anthropic" and not self.anthropic_api_key:
            missing.append("ANTHROPIC_API_KEY")
        if self.llm_provider == "groq" and not self.groq_api_key:
            missing.append("GROQ_API_KEY")
        if self.stt_provider == "google" and not self.google_speech_api_key:
            missing.append("GOOGLE_SPEECH_API_KEY")
        if self.stt_provider == "deepgram" and not self.deepgram_api_key:
            missing.append("DEEPGRAM_API_KEY")
        if not self.elevenlabs_api_key:
            missing.append("ELEVENLABS_API_KEY")
        if not self.elevenlabs_voice_id:
            missing.append("ELEVENLABS_VOICE_ID")
        if not self.twilio_account_sid:
            missing.append("TWILIO_ACCOUNT_SID")
        if not self.twilio_auth_token:
            missing.append("TWILIO_AUTH_TOKEN")
        if not self.twilio_phone_number:
            missing.append("TWILIO_PHONE_NUMBER")
        if not self.normalized_base_url():
            missing.append("BASE_URL")
        return missing

    def telephony_ready(self) -> bool:
        return not self.missing_runtime_values()

    def config_manager_kind(self) -> str:
        if self.use_redis_config_manager and self.redis_url:
            return "redis"
        return "memory"

    def provider_summary(self) -> dict[str, str]:
        return {
            "llm_provider": self.llm_provider,
            "stt_provider": self.stt_provider,
            "tts_provider": "elevenlabs",
            "tts_transport": "http_stream",
            "calendar": "stub",
            "email": "stub",
            "crm": "stub",
            "persistence": "memory-first",
        }


@lru_cache(maxsize=1)
def get_settings() -> SDRSettings:
    return SDRSettings()
