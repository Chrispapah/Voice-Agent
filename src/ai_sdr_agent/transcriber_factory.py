from __future__ import annotations

import asyncio
import re

from loguru import logger
from vocode.streaming.models.transcriber import DeepgramTranscriberConfig, TranscriberConfig, Transcription
from vocode.streaming.transcriber.deepgram_transcriber import DeepgramTranscriber
from vocode.streaming.transcriber.default_factory import DefaultTranscriberFactory

from ai_sdr_agent.services.latency_analytics import mark_deepgram_final_transcript_enqueued_from_context

_ENUM_PATTERN = re.compile(r"SamplingRate\.RATE_(\d+)")


class _TranscriberOutputQueueProxy:
    """Wraps the transcriber output queue to timestamp final Deepgram transcripts."""

    __slots__ = ("_inner",)

    def __init__(self, inner: asyncio.Queue) -> None:
        self._inner = inner

    def put_nowait(self, item) -> None:
        if isinstance(item, Transcription) and item.is_final:
            mark_deepgram_final_transcript_enqueued_from_context()
        return self._inner.put_nowait(item)

    def __getattr__(self, name: str):
        return getattr(self._inner, name)


class LoggingDeepgramTranscriber(DeepgramTranscriber):
    """Patches the Deepgram URL to fix enum serialization bugs in Vocode,
    and adds startup/crash logging for telephony diagnostics."""

    def __init__(self, transcriber_config: DeepgramTranscriberConfig):
        super().__init__(transcriber_config)
        self.output_queue = _TranscriberOutputQueueProxy(self.output_queue)

    def get_deepgram_url(self) -> str:
        url = super().get_deepgram_url()
        url = _ENUM_PATTERN.sub(r"\1", url)
        return url

    async def _run_loop(self):
        url = self.get_deepgram_url()
        logger.info("Deepgram WS connecting url={}", url)
        try:
            return await super()._run_loop()
        except Exception:
            logger.exception("Deepgram transcriber loop crashed")
            raise


class SDRTranscriberFactory(DefaultTranscriberFactory):
    def create_transcriber(
        self,
        transcriber_config: TranscriberConfig,
    ):
        if isinstance(transcriber_config, DeepgramTranscriberConfig):
            return LoggingDeepgramTranscriber(transcriber_config)
        return super().create_transcriber(transcriber_config)

