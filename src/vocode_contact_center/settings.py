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
    redis_url: str | None = None

    deepgram_api_key: str | None = None
    deepgram_model: str = "nova-2"

    elevenlabs_api_key: str | None = None
    elevenlabs_voice_id: str | None = None
    elevenlabs_model_id: str = "eleven_turbo_v2"
    elevenlabs_use_websocket: bool = False

    langchain_provider: str = "openai"
    langchain_model_name: str = "gpt-4o-mini"
    langchain_temperature: float = 0.2
    langchain_max_tokens: int = 256

    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    google_api_key: str | None = None

    agent_name: str = "AI Contact Center"
    agent_initial_message: str = "Thanks for calling. How can I help you today?"
    agent_prompt_preamble: str = DEFAULT_AGENT_PROMPT
    transfer_phone_number: str | None = None

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
