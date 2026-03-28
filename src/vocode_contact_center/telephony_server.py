from __future__ import annotations

import asyncio
import threading
from typing import AsyncGenerator, Optional

from fastapi import APIRouter
from loguru import logger
from vocode.streaming.agent.abstract_factory import AbstractAgentFactory
from vocode.streaming.agent.default_factory import DefaultAgentFactory
from vocode.streaming.models.telephony import BaseCallConfig, TwilioCallConfig
from vocode.streaming.models.transcriber import Transcription
from vocode.streaming.synthesizer.abstract_factory import AbstractSynthesizerFactory
from vocode.streaming.synthesizer.base_synthesizer import SynthesisResult
from vocode.streaming.synthesizer.default_factory import DefaultSynthesizerFactory
from vocode.streaming.telephony.config_manager.base_config_manager import BaseConfigManager
from vocode.streaming.telephony.conversation.twilio_phone_conversation import (
    TwilioPhoneConversation,
)
from vocode.streaming.telephony.server.base import (
    AbstractInboundCallConfig,
    TelephonyServer,
)
from vocode.streaming.telephony.server.router.calls import CallsRouter
from vocode.streaming.transcriber.abstract_factory import AbstractTranscriberFactory
from vocode.streaming.transcriber.default_factory import DefaultTranscriberFactory
from vocode.streaming.utils.events_manager import EventsManager

from vocode_contact_center.latency_tracker import conversation_latency_tracker


class LatencyTrackingTwilioPhoneConversation(TwilioPhoneConversation):
    class TranscriptionsWorker(TwilioPhoneConversation.TranscriptionsWorker):
        async def process(self, transcription: Transcription):
            if transcription.is_final:
                conversation_latency_tracker.mark_transcription_final(
                    self.conversation.id,
                    transcription.message,
                )
            await super().process(transcription)

    def receive_audio(self, chunk: bytes):
        conversation_latency_tracker.mark_audio_received(self.id)
        super().receive_audio(chunk)

    async def _send_chunks(
        self,
        utterance_id: str,
        chunk_generator: AsyncGenerator[SynthesisResult.ChunkResult, None],
        clear_message_lock: asyncio.Lock,
        stop_event: threading.Event,
    ):
        chunk_idx = 0
        first_chunk_sent = False
        try:
            async for chunk_result in chunk_generator:
                async with clear_message_lock:
                    if stop_event.is_set():
                        break
                    if not first_chunk_sent:
                        conversation_latency_tracker.mark_first_tts_chunk(self.id)
                        first_chunk_sent = True
                    self.output_device.consume_nonblocking(chunk_result.chunk)
                    self.output_device.send_chunk_finished_mark(utterance_id, chunk_idx)
                    chunk_idx += 1
        except asyncio.CancelledError:
            pass
        finally:
            logger.debug("Finished sending all chunks to Twilio")
            self.output_device.send_utterance_finished_mark(utterance_id)

    async def terminate(self):
        try:
            await super().terminate()
        finally:
            conversation_latency_tracker.clear(self.id)


class LatencyTrackingCallsRouter(CallsRouter):
    def _from_call_config(
        self,
        base_url: str,
        call_config: BaseCallConfig,
        config_manager: BaseConfigManager,
        conversation_id: str,
        transcriber_factory: AbstractTranscriberFactory = DefaultTranscriberFactory(),
        agent_factory: AbstractAgentFactory = DefaultAgentFactory(),
        synthesizer_factory: AbstractSynthesizerFactory = DefaultSynthesizerFactory(),
        events_manager: Optional[EventsManager] = None,
    ):
        if isinstance(call_config, TwilioCallConfig):
            return LatencyTrackingTwilioPhoneConversation(
                to_phone=call_config.to_phone,
                from_phone=call_config.from_phone,
                base_url=base_url,
                config_manager=config_manager,
                agent_config=call_config.agent_config,
                transcriber_config=call_config.transcriber_config,
                synthesizer_config=call_config.synthesizer_config,
                twilio_config=call_config.twilio_config,
                twilio_sid=call_config.twilio_sid,
                conversation_id=conversation_id,
                transcriber_factory=transcriber_factory,
                agent_factory=agent_factory,
                synthesizer_factory=synthesizer_factory,
                events_manager=events_manager,
                direction=call_config.direction,
            )
        return super()._from_call_config(
            base_url=base_url,
            call_config=call_config,
            config_manager=config_manager,
            conversation_id=conversation_id,
            transcriber_factory=transcriber_factory,
            agent_factory=agent_factory,
            synthesizer_factory=synthesizer_factory,
            events_manager=events_manager,
        )


class LatencyTrackingTelephonyServer(TelephonyServer):
    def __init__(
        self,
        base_url: str,
        config_manager: BaseConfigManager,
        inbound_call_configs: list[AbstractInboundCallConfig] | None = None,
        transcriber_factory: AbstractTranscriberFactory = DefaultTranscriberFactory(),
        agent_factory: AbstractAgentFactory = DefaultAgentFactory(),
        synthesizer_factory: AbstractSynthesizerFactory = DefaultSynthesizerFactory(),
        events_manager: Optional[EventsManager] = None,
    ):
        self.base_url = base_url
        self.router = APIRouter()
        self.config_manager = config_manager
        self.events_manager = events_manager
        self.router.include_router(
            LatencyTrackingCallsRouter(
                base_url=base_url,
                config_manager=self.config_manager,
                transcriber_factory=transcriber_factory,
                agent_factory=agent_factory,
                synthesizer_factory=synthesizer_factory,
                events_manager=self.events_manager,
            ).get_router()
        )
        for config in inbound_call_configs or []:
            self.router.add_api_route(
                config.url,
                self.create_inbound_route(inbound_call_config=config),
                methods=["POST"],
            )
        self.router.add_api_route("/events", self.events, methods=["GET", "POST"])
        logger.info("Set up events endpoint at https://{}/events", self.base_url)

        self.router.add_api_route(
            "/recordings/{conversation_id}",
            self.recordings,
            methods=["GET", "POST"],
        )
        logger.info(
            "Set up recordings endpoint at https://{}/recordings/{{conversation_id}}",
            self.base_url,
        )
