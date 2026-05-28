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

    app_name: str = "Voice Agent"
    app_env: str = "development"
    host: str = "0.0.0.0"
    port: int = 3000
    base_url: str | None = None

    llm_provider: Literal["openai", "anthropic", "groq", "stub"] = "groq"
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
    google_speech_language_code: str = "el-GR"
    # phone_call + enhanced matches Twilio mulaw telephony (see google_speech_transcriber).
    google_speech_model: str | None = "phone_call"
    # Client-side: silence after last interim before a "final" (lower = snappier, more mid-sentence cuts). Floored in code at 40ms.
    google_utterance_silence_ms: int = 300
    google_mute_during_speech: bool = True

    deepgram_api_key: str | None = None
    deepgram_language: str = "el"
    deepgram_model: str = "nova-2"
    # Endpointing: lower = final transcript sooner after a pause (more false end-of-turn / mid-sentence cuts).
    # This value drives (1) Vocode vad_threshold_ms on Twilio and (2) Deepgram Live ``endpointing`` (ms) for browser voice.
    # time/punctuation = Vocode fallbacks when speech_final lags.
    # utterance_cutoff_ms → Deepgram utterance_end_ms (Vocode enforces ≥1000). Higher = more tolerance for
    # pauses within a thought; lower = snappier turns at the cost of mid-phrase cuts.
    deepgram_vad_threshold_ms: int = 55
    deepgram_utterance_cutoff_ms: int = 1500
    deepgram_time_cutoff_seconds: float = 0.05
    deepgram_post_punctuation_time_seconds: float = 0.02
    deepgram_single_utterance_for_first_response: bool = True
    deepgram_mute_during_speech: bool = True

    tts_provider: Literal["elevenlabs", "azure"] = "elevenlabs"
    elevenlabs_api_key: str | None = None
    elevenlabs_voice_id: str | None = "4hx4668A4ljDTKS4m5oV"
    elevenlabs_model_id: str = "eleven_turbo_v2"
    elevenlabs_use_websocket: bool = False
    elevenlabs_optimize_streaming_latency: int = 4

    voice_provider: Literal["builtin", "openai_realtime", "openai_realtime_elevenlabs"] = "builtin"
    openai_realtime_model: str = "gpt-realtime"
    openai_realtime_voice: str = "alloy"
    openai_realtime_instructions: str | None = None
    openai_realtime_transcription_model: str = "gpt-4o-mini-transcribe"
    # Higher VAD threshold and longer silence reduce false barge-ins from speaker echo/noise.
    openai_realtime_vad_threshold: float = 0.75
    openai_realtime_vad_silence_duration_ms: int = 700
    openai_realtime_vad_prefix_padding_ms: int = 300

    azure_speech_key: str | None = None
    azure_speech_region: str | None = None
    azure_voice_name: str = "el-GR-AthinaNeural"

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
    max_call_turns: int = 12

    def normalized_base_url(self) -> str | None:
        if not self.base_url:
            return None
        value = self.base_url.strip()
        if value.startswith("http://"):
            value = value[len("http://") :]
        if value.startswith("https://"):
            value = value[len("https://") :]
        return value.rstrip("/")

    def provider_summary(self) -> dict[str, str]:
        return {
            "llm_provider": self.llm_provider,
            "voice_provider": self.voice_provider,
            "tts_provider": "openai_realtime" if self.voice_provider == "openai_realtime" else "elevenlabs",
            "persistence": "postgres",
        }


@lru_cache(maxsize=1)
def get_settings() -> SDRSettings:
    return SDRSettings()
