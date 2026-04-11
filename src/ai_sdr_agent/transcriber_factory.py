from __future__ import annotations

from loguru import logger
from vocode.streaming.models.transcriber import DeepgramTranscriberConfig, TranscriberConfig
from vocode.streaming.transcriber.deepgram_transcriber import DeepgramTranscriber
from vocode.streaming.transcriber.default_factory import DefaultTranscriberFactory


class LoggingDeepgramTranscriber(DeepgramTranscriber):
    """Deepgram transcriber with startup logging for telephony diagnostics."""

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

