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
from ai_sdr_agent.vocode_agent import build_agent_config


@dataclass
class ScheduledCallResult:
    conversation_id: str
    lead_id: str
    status: str


class CallScheduler:
    def __init__(
        self,
        *,
        settings: SDRSettings,
        config_manager: BaseConfigManager,
    ):
        self.settings = settings
        self.config_manager = config_manager

    async def schedule_outbound_call(self, lead: LeadRecord) -> ScheduledCallResult:
        if not self.settings.telephony_ready():
            raise RuntimeError(
                "Telephony is not configured. Missing: "
                + ", ".join(self.settings.missing_runtime_values())
            )
        if not self.settings.normalized_base_url():
            raise RuntimeError("BASE_URL is required for outbound calls.")
        agent_config = build_agent_config(
            lead_id=lead.lead_id,
            calendar_id=lead.calendar_id,
            sales_rep_name=lead.owner_name,
            initial_message_text=self.settings.initial_greeting,
        )
        outbound_call = OutboundCall(
            base_url=self.settings.normalized_base_url(),
            to_phone=lead.phone_number,
            from_phone=self.settings.twilio_phone_number or "",
            config_manager=self.config_manager,
            agent_config=agent_config,
            telephony_config=TwilioConfig(
                account_sid=self.settings.twilio_account_sid or "",
                auth_token=self.settings.twilio_auth_token or "",
                record=self.settings.twilio_record_calls,
            ),
            transcriber_config=DeepgramTranscriberConfig.from_telephone_input_device(
                endpointing_config=DeepgramEndpointingConfig(
                    vad_threshold_ms=self.settings.deepgram_vad_threshold_ms,
                    utterance_cutoff_ms=self.settings.deepgram_utterance_cutoff_ms,
                    time_silent_config=TimeSilentConfig(
                        time_cutoff_seconds=self.settings.deepgram_time_cutoff_seconds,
                        post_punctuation_time_seconds=self.settings.deepgram_post_punctuation_time_seconds,
                    ),
                    use_single_utterance_endpointing_for_first_utterance=(
                        self.settings.deepgram_single_utterance_for_first_response
                    ),
                ),
                api_key=self.settings.deepgram_api_key,
                model=self.settings.deepgram_model,
                mute_during_speech=self.settings.deepgram_mute_during_speech,
            ),
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
            api_key=self.settings.elevenlabs_api_key or "",
            voice_id=self.settings.elevenlabs_voice_id or "",
            model_id=self.settings.elevenlabs_model_id,
            optimize_streaming_latency=self.settings.elevenlabs_optimize_streaming_latency,
            experimental_websocket=self.settings.elevenlabs_use_websocket,
        )
