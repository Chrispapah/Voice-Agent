from __future__ import annotations

import re

from loguru import logger
from vocode.streaming.models.transcriber import DeepgramTranscriberConfig, TranscriberConfig
from vocode.streaming.transcriber.deepgram_transcriber import DeepgramTranscriber
from vocode.streaming.transcriber.default_factory import DefaultTranscriberFactory

_ENUM_PATTERN = re.compile(r"SamplingRate\.RATE_(\d+)")


class LoggingDeepgramTranscriber(DeepgramTranscriber):
    """Patches the Deepgram URL to fix enum serialization bugs in Vocode,
    and adds startup/crash logging for telephony diagnostics."""

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

