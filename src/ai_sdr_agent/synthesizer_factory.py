"""ElevenLabs synthesizers with first-chunk perceived-latency hooks (phone)."""

from __future__ import annotations

from typing import AsyncGenerator

from vocode.streaming.models.message import BaseMessage, SilenceMessage
from vocode.streaming.models.synthesizer import (
    ElevenLabsSynthesizerConfig,
    SynthesizerConfig,
)
from vocode.streaming.synthesizer.base_synthesizer import BaseSynthesizer, SynthesisResult
from vocode.streaming.synthesizer.default_factory import DefaultSynthesizerFactory
from vocode.streaming.synthesizer.eleven_labs_synthesizer import ElevenLabsSynthesizer
from vocode.streaming.synthesizer.eleven_labs_websocket_synthesizer import ElevenLabsWSSynthesizer

from ai_sdr_agent.services.latency_analytics import note_first_tts_audio_chunk_from_context


async def _wrap_synthesis_chunk_generator(
    inner: AsyncGenerator[SynthesisResult.ChunkResult, None],
) -> AsyncGenerator[SynthesisResult.ChunkResult, None]:
    first = True
    async for chunk_result in inner:
        if first and chunk_result.chunk:
            note_first_tts_audio_chunk_from_context()
            first = False
        yield chunk_result


class InstrumentedElevenLabsSynthesizer(ElevenLabsSynthesizer):
    async def create_speech(
        self,
        message: BaseMessage,
        chunk_size: int,
        is_first_text_chunk: bool = False,
        is_sole_text_chunk: bool = False,
    ) -> SynthesisResult:
        if isinstance(message, SilenceMessage):
            return await super().create_speech(
                message,
                chunk_size,
                is_first_text_chunk=is_first_text_chunk,
                is_sole_text_chunk=is_sole_text_chunk,
            )
        result = await super().create_speech(
            message,
            chunk_size,
            is_first_text_chunk=is_first_text_chunk,
            is_sole_text_chunk=is_sole_text_chunk,
        )
        result.chunk_generator = _wrap_synthesis_chunk_generator(result.chunk_generator)
        return result


class InstrumentedElevenLabsWSSynthesizer(ElevenLabsWSSynthesizer):
    async def create_speech(
        self,
        message: BaseMessage,
        chunk_size: int,
        is_first_text_chunk: bool = False,
        is_sole_text_chunk: bool = False,
    ) -> SynthesisResult:
        if isinstance(message, SilenceMessage):
            return await super().create_speech(
                message,
                chunk_size,
                is_first_text_chunk=is_first_text_chunk,
                is_sole_text_chunk=is_sole_text_chunk,
            )
        result = await super().create_speech(
            message,
            chunk_size,
            is_first_text_chunk=is_first_text_chunk,
            is_sole_text_chunk=is_sole_text_chunk,
        )
        result.chunk_generator = _wrap_synthesis_chunk_generator(result.chunk_generator)
        return result


class SDRSynthesizerFactory(DefaultSynthesizerFactory):
    """Use instrumented ElevenLabs classes so we log perceived (graph → first audio) latency."""

    def create_synthesizer(
        self,
        synthesizer_config: SynthesizerConfig,
    ) -> BaseSynthesizer:
        if isinstance(synthesizer_config, ElevenLabsSynthesizerConfig):
            if synthesizer_config.experimental_websocket:
                return InstrumentedElevenLabsWSSynthesizer(synthesizer_config)
            return InstrumentedElevenLabsSynthesizer(synthesizer_config)
        return super().create_synthesizer(synthesizer_config)
