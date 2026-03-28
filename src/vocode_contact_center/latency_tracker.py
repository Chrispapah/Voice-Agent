from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from time import perf_counter

from loguru import logger


@dataclass
class ConversationLatencyState:
    turn_index: int = 0
    last_audio_received_at: float | None = None
    transcription_final_at: float | None = None
    first_llm_token_at: float | None = None
    first_tts_chunk_at: float | None = None


class ConversationLatencyTracker:
    def __init__(self) -> None:
        self._lock = Lock()
        self._states: dict[str, ConversationLatencyState] = {}

    def _get_state(self, conversation_id: str) -> ConversationLatencyState:
        state = self._states.get(conversation_id)
        if state is None:
            state = ConversationLatencyState()
            self._states[conversation_id] = state
        return state

    def mark_audio_received(self, conversation_id: str) -> None:
        now = perf_counter()
        with self._lock:
            state = self._get_state(conversation_id)
            state.last_audio_received_at = now

    def mark_transcription_final(self, conversation_id: str, transcript: str) -> None:
        now = perf_counter()
        with self._lock:
            state = self._get_state(conversation_id)
            state.turn_index += 1
            state.transcription_final_at = now
            state.first_llm_token_at = None
            state.first_tts_chunk_at = None

            audio_to_final_ms: float | None = None
            if state.last_audio_received_at is not None:
                audio_to_final_ms = (now - state.last_audio_received_at) * 1000

        logger.info(
            "Latency turn {} for {}: last_audio->final_transcript={} ms transcript={!r}",
            state.turn_index,
            conversation_id,
            f"{audio_to_final_ms:.0f}" if audio_to_final_ms is not None else "n/a",
            transcript[:80],
        )

    def mark_first_llm_token(
        self,
        conversation_id: str,
        *,
        using_input_streaming_synthesizer: bool,
    ) -> None:
        now = perf_counter()
        with self._lock:
            state = self._get_state(conversation_id)
            if state.first_llm_token_at is not None:
                return

            state.first_llm_token_at = now

            final_to_llm_ms: float | None = None
            if state.transcription_final_at is not None:
                final_to_llm_ms = (now - state.transcription_final_at) * 1000

        logger.info(
            "Latency turn {} for {}: final_transcript->first_llm_token={} ms input_streaming_synthesizer={}",
            state.turn_index,
            conversation_id,
            f"{final_to_llm_ms:.0f}" if final_to_llm_ms is not None else "n/a",
            using_input_streaming_synthesizer,
        )

    def mark_first_tts_chunk(self, conversation_id: str) -> None:
        now = perf_counter()
        with self._lock:
            state = self._get_state(conversation_id)
            if state.first_tts_chunk_at is not None:
                return

            state.first_tts_chunk_at = now

            llm_to_tts_ms: float | None = None
            total_turn_ms: float | None = None
            if state.first_llm_token_at is not None:
                llm_to_tts_ms = (now - state.first_llm_token_at) * 1000
            if state.transcription_final_at is not None:
                total_turn_ms = (now - state.transcription_final_at) * 1000

        logger.info(
            "Latency turn {} for {}: first_llm_token->first_tts_chunk={} ms turn_final->first_tts_chunk={} ms",
            state.turn_index,
            conversation_id,
            f"{llm_to_tts_ms:.0f}" if llm_to_tts_ms is not None else "n/a",
            f"{total_turn_ms:.0f}" if total_turn_ms is not None else "n/a",
        )

    def clear(self, conversation_id: str) -> None:
        with self._lock:
            self._states.pop(conversation_id, None)


conversation_latency_tracker = ConversationLatencyTracker()
