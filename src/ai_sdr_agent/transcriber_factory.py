from __future__ import annotations

import asyncio
import re
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from loguru import logger
from vocode.streaming.models.transcriber import DeepgramTranscriberConfig, TranscriberConfig, Transcription
from vocode.streaming.transcriber.deepgram_transcriber import (
    DeepgramEndpointingConfig,
    DeepgramTranscriber,
    TimeSilentConfig,
)
from vocode.streaming.transcriber.default_factory import DefaultTranscriberFactory

from ai_sdr_agent.config import SDRSettings
from ai_sdr_agent.services.latency_analytics import (
    mark_deepgram_final_transcript_enqueued_from_context,
    mark_last_inbound_audio_from_context,
)

_ENUM_PATTERN = re.compile(r"SamplingRate\.RATE_(\d+)")


def _rewrite_deepgram_utterance_end_ms_url(url: str, transcriber: DeepgramTranscriber) -> str:
    """Vocode uses max(utterance_cutoff_ms, 1000) for utterance_end_ms; override with configured ms."""
    ep = transcriber.transcriber_config.endpointing_config
    if not isinstance(ep, DeepgramEndpointingConfig):
        return url
    parsed = urlparse(url)
    pairs = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True) if k != "utterance_end_ms"]
    coef = transcriber._get_speed_coefficient()
    ut_ms = max(1, int(ep.utterance_cutoff_ms * (1 / coef)))
    pairs.append(("utterance_end_ms", str(ut_ms)))
    new_query = urlencode(pairs)
    return urlunparse(parsed._replace(query=new_query))


def resolve_telephony_deepgram_model(raw_model: str | None) -> str:
    """Twilio sends 8-bit mulaw; prefer telephony-tuned Deepgram models."""
    model = (raw_model or "").strip() or "phonecall"
    if model.lower() == "nova-2":
        logger.warning(
            "DEEPGRAM_MODEL={} is not telephony-optimized; using phonecall model for Twilio audio.",
            model,
        )
        return "phonecall"
    return model


def build_telephony_deepgram_transcriber_config(settings: SDRSettings) -> DeepgramTranscriberConfig:
    """Deepgram VAD + silence/punctuation fallback (matches outbound scheduler when using SDRSettings)."""
    model = resolve_telephony_deepgram_model(settings.deepgram_model)
    # Floors keep extreme env values from breaking STT; still allow aggressive sub-100ms VAD via vad_threshold_ms.
    time_cutoff = max(settings.deepgram_time_cutoff_seconds, 0.04)
    post_punct = max(settings.deepgram_post_punctuation_time_seconds, 0.02)
    logger.info(
        "Deepgram telephony endpointing model={} vad_ms={} utterance_cutoff_ms={} "
        "time_silent_cutoff={}s post_punctuation={}s",
        model,
        settings.deepgram_vad_threshold_ms,
        settings.deepgram_utterance_cutoff_ms,
        time_cutoff,
        post_punct,
    )
    return DeepgramTranscriberConfig.from_telephone_input_device(
        endpointing_config=DeepgramEndpointingConfig(
            vad_threshold_ms=settings.deepgram_vad_threshold_ms,
            utterance_cutoff_ms=settings.deepgram_utterance_cutoff_ms,
            time_silent_config=TimeSilentConfig(
                time_cutoff_seconds=time_cutoff,
                post_punctuation_time_seconds=post_punct,
            ),
            use_single_utterance_endpointing_for_first_utterance=settings.deepgram_single_utterance_for_first_response,
        ),
        api_key=settings.deepgram_api_key,
        language=settings.deepgram_language,
        model=model,
        mute_during_speech=settings.deepgram_mute_during_speech,
    )


def _chunk_is_inbound_audio(item: object) -> bool:
    """True for raw telephony audio; false for vocode Deepgram CloseStream JSON."""
    if not isinstance(item, (bytes, bytearray)):
        return False
    if len(item) == 0:
        return False
    if item.startswith(b"{") and b"CloseStream" in item:
        return False
    return True


class _TranscriberInputQueueProxy:
    """Timestamps last raw inbound audio chunk per call (for last_audio → STT final metrics)."""

    __slots__ = ("_inner",)

    def __init__(self, inner: asyncio.Queue) -> None:
        self._inner = inner

    def put_nowait(self, item) -> None:
        if _chunk_is_inbound_audio(item):
            mark_last_inbound_audio_from_context()
        return self._inner.put_nowait(item)

    def __getattr__(self, name: str):
        return getattr(self._inner, name)


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
        self.input_queue = _TranscriberInputQueueProxy(self.input_queue)
        self.output_queue = _TranscriberOutputQueueProxy(self.output_queue)

    def get_deepgram_url(self) -> str:
        url = super().get_deepgram_url()
        url = _ENUM_PATTERN.sub(r"\1", url)
        url = _rewrite_deepgram_utterance_end_ms_url(url, self)
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

