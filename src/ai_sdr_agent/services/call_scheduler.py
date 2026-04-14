from __future__ import annotations

from dataclasses import dataclass

from loguru import logger
from vocode.streaming.models.synthesizer import ElevenLabsSynthesizerConfig
from vocode.streaming.models.telephony import TwilioConfig
from vocode.streaming.models.transcriber import DeepgramTranscriberConfig
from vocode.streaming.telephony.config_manager.base_config_manager import BaseConfigManager
from vocode.streaming.telephony.conversation.outbound_call import OutboundCall
from vocode.streaming.transcriber.deepgram_transcriber import (
    DeepgramEndpointingConfig,
    TimeSilentConfig,
)

from ai_sdr_agent.config import SDRSettings
from ai_sdr_agent.models import LeadRecord
from ai_sdr_agent.transcriber_factory import (
    build_telephony_deepgram_transcriber_config,
    resolve_telephony_deepgram_model,
)
from ai_sdr_agent.vocode_agent import build_agent_config


@dataclass
class ScheduledCallResult:
    conversation_id: str
    lead_id: str
    status: str


class CallScheduler:
    """Schedule outbound calls.  Accepts either SDRSettings (legacy) or per-bot config dict."""

    def __init__(
        self,
        *,
        settings: SDRSettings | None = None,
        config_manager: BaseConfigManager,
        bot_config: dict | None = None,
    ):
        self.settings = settings
        self.config_manager = config_manager
        self._bot_config = bot_config

    def _cfg(self, key: str, default=None):
        if self._bot_config:
            return self._bot_config.get(key, default)
        if self.settings:
            return getattr(self.settings, key, default)
        return default

    def _telephony_ready(self) -> bool:
        required = [
            self._cfg("deepgram_api_key"),
            self._cfg("elevenlabs_api_key"),
            self._cfg("elevenlabs_voice_id"),
            self._cfg("twilio_account_sid"),
            self._cfg("twilio_auth_token"),
            self._cfg("twilio_phone_number"),
        ]
        base_url = self.settings.normalized_base_url() if self.settings else self._cfg("base_url")
        return all(required) and bool(base_url)

    def _base_url(self) -> str:
        if self.settings:
            return self.settings.normalized_base_url() or ""
        return self._cfg("base_url", "")

    async def schedule_outbound_call(self, lead: LeadRecord) -> ScheduledCallResult:
        if not self._telephony_ready():
            raise RuntimeError("Telephony is not configured for this bot.")
        base_url = self._base_url()
        if not base_url:
            raise RuntimeError("BASE_URL is required for outbound calls.")

        agent_config = build_agent_config(
            lead_id=lead.lead_id,
            calendar_id=lead.calendar_id,
            sales_rep_name=lead.owner_name,
            initial_message_text=self._cfg(
                "initial_greeting",
                "Hi, this is your AI assistant. Do you have a moment?",
            ),
        )
        if self.settings is not None and not self._bot_config:
            transcriber_config = build_telephony_deepgram_transcriber_config(self.settings)
        else:
            transcriber_config = DeepgramTranscriberConfig.from_telephone_input_device(
                endpointing_config=DeepgramEndpointingConfig(
                    vad_threshold_ms=self._cfg("deepgram_vad_threshold_ms", 120),
                    utterance_cutoff_ms=self._cfg("deepgram_utterance_cutoff_ms", 900),
                    time_silent_config=TimeSilentConfig(
                        time_cutoff_seconds=max(
                            self._cfg("deepgram_time_cutoff_seconds", 0.12), 0.05
                        ),
                        post_punctuation_time_seconds=max(
                            self._cfg("deepgram_post_punctuation_time_seconds", 0.05), 0.03
                        ),
                    ),
                    use_single_utterance_endpointing_for_first_utterance=self._cfg(
                        "deepgram_single_utterance_for_first_response", True
                    ),
                ),
                api_key=self._cfg("deepgram_api_key"),
                language=self._cfg("deepgram_language", "en-US"),
                model=resolve_telephony_deepgram_model(self._cfg("deepgram_model", "nova-2")),
                mute_during_speech=self._cfg("deepgram_mute_during_speech", True),
            )
        outbound_call = OutboundCall(
            base_url=base_url,
            to_phone=lead.phone_number,
            from_phone=self._cfg("twilio_phone_number", ""),
            config_manager=self.config_manager,
            agent_config=agent_config,
            telephony_config=TwilioConfig(
                account_sid=self._cfg("twilio_account_sid", ""),
                auth_token=self._cfg("twilio_auth_token", ""),
                record=self._cfg("twilio_record_calls", False),
            ),
            transcriber_config=transcriber_config,
            synthesizer_config=self._build_synthesizer_config(),
            telephony_params={
                "lead_id": lead.lead_id,
                "calendar_id": lead.calendar_id,
            },
        )
        await outbound_call.start()
        logger.info(
            "Outbound SDR call started lead_id={} conversation_id={}",
            lead.lead_id,
            outbound_call.conversation_id,
        )
        return ScheduledCallResult(
            conversation_id=outbound_call.conversation_id,
            lead_id=lead.lead_id,
            status="started",
        )

    def _build_synthesizer_config(self):
        return ElevenLabsSynthesizerConfig.from_telephone_output_device(
            api_key=self._cfg("elevenlabs_api_key", ""),
            voice_id=self._cfg("elevenlabs_voice_id", ""),
            model_id=self._cfg("elevenlabs_model_id", "eleven_turbo_v2"),
            optimize_streaming_latency=self._cfg("elevenlabs_optimize_streaming_latency", 4),
            experimental_websocket=self._cfg("elevenlabs_use_websocket", False),
        )
