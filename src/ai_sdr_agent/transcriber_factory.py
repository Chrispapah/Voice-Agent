from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from vocode.streaming.models.transcriber import DeepgramTranscriberConfig, TranscriberConfig
from vocode.streaming.transcriber.deepgram_transcriber import DeepgramTranscriber
from vocode.streaming.transcriber.default_factory import DefaultTranscriberFactory


class StableDeepgramTranscriber(DeepgramTranscriber):
    """Force final-only Deepgram results for short telephony utterances."""

    def get_deepgram_url(self) -> str:
        base_url = super().get_deepgram_url()
        parts = urlsplit(base_url)
        params = dict(parse_qsl(parts.query, keep_blank_values=True))
        params["interim_results"] = "false"
        return urlunsplit(
            (parts.scheme, parts.netloc, parts.path, urlencode(params), parts.fragment)
        )


class SDRTranscriberFactory(DefaultTranscriberFactory):
    def create_transcriber(
        self,
        transcriber_config: TranscriberConfig,
    ):
        if isinstance(transcriber_config, DeepgramTranscriberConfig):
            return StableDeepgramTranscriber(transcriber_config)
        return super().create_transcriber(transcriber_config)

