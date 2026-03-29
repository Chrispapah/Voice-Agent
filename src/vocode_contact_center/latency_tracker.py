from __future__ import annotations

from collections import deque
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


@dataclass
class LatencyAggregate:
    count: int = 0
    total_ms: float = 0.0
    min_ms: float | None = None
    max_ms: float | None = None
    last_ms: float | None = None

    def record(self, latency_ms: float) -> None:
        self.count += 1
        self.total_ms += latency_ms
        self.last_ms = latency_ms
        self.min_ms = latency_ms if self.min_ms is None else min(self.min_ms, latency_ms)
        self.max_ms = latency_ms if self.max_ms is None else max(self.max_ms, latency_ms)

    def snapshot(self) -> dict[str, float | int | None]:
        average_ms = None
        if self.count:
            average_ms = round(self.total_ms / self.count, 1)
        return {
            "count": self.count,
            "avg_ms": average_ms,
            "min_ms": round(self.min_ms, 1) if self.min_ms is not None else None,
            "max_ms": round(self.max_ms, 1) if self.max_ms is not None else None,
            "last_ms": round(self.last_ms, 1) if self.last_ms is not None else None,
        }


class ConversationLatencyTracker:
    def __init__(self) -> None:
        self._lock = Lock()
        self._states: dict[str, ConversationLatencyState] = {}
        self._aggregates = {
            "audio_to_final_ms": LatencyAggregate(),
            "final_to_first_llm_ms": LatencyAggregate(),
            "first_llm_to_tts_ms": LatencyAggregate(),
            "final_to_first_tts_ms": LatencyAggregate(),
        }
        self._recent_turns: deque[dict[str, float | int | str | None]] = deque(maxlen=50)

    def _get_state(self, conversation_id: str) -> ConversationLatencyState:
        state = self._states.get(conversation_id)
        if state is None:
            state = ConversationLatencyState()
            self._states[conversation_id] = state
        return state

    def _record_aggregate(
        self,
        metric_name: str,
        latency_ms: float | None,
        *,
        conversation_id: str,
        turn_index: int,
    ) -> None:
        if latency_ms is None:
            return
        aggregate = self._aggregates[metric_name]
        aggregate.record(latency_ms)
        self._recent_turns.append(
            {
                "conversation_id": conversation_id,
                "turn_index": turn_index,
                "metric": metric_name,
                "latency_ms": round(latency_ms, 1),
            }
        )

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
            self._record_aggregate(
                "audio_to_final_ms",
                audio_to_final_ms,
                conversation_id=conversation_id,
                turn_index=state.turn_index,
            )

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
            self._record_aggregate(
                "final_to_first_llm_ms",
                final_to_llm_ms,
                conversation_id=conversation_id,
                turn_index=state.turn_index,
            )

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
            self._record_aggregate(
                "first_llm_to_tts_ms",
                llm_to_tts_ms,
                conversation_id=conversation_id,
                turn_index=state.turn_index,
            )
            self._record_aggregate(
                "final_to_first_tts_ms",
                total_turn_ms,
                conversation_id=conversation_id,
                turn_index=state.turn_index,
            )

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

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "active_conversations": len(self._states),
                "segments": {
                    metric_name: aggregate.snapshot()
                    for metric_name, aggregate in self._aggregates.items()
                },
                "recent_turns": list(self._recent_turns),
            }


conversation_latency_tracker = ConversationLatencyTracker()
