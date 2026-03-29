from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from threading import Lock
from time import perf_counter

from loguru import logger


@dataclass
class ConversationLatencyState:
    turn_index: int = 0
    started_at: float | None = None
    from_phone: str | None = None
    to_phone: str | None = None
    last_audio_received_at: float | None = None
    transcription_final_at: float | None = None
    first_model_token_at: float | None = None
    first_llm_token_at: float | None = None
    first_tts_chunk_at: float | None = None
    last_transcript_preview: str | None = None
    using_input_streaming_synthesizer: bool | None = None
    audio_to_final_aggregate: LatencyAggregate = None  # type: ignore[assignment]
    final_to_model_aggregate: LatencyAggregate = None  # type: ignore[assignment]
    final_to_response_chunk_aggregate: LatencyAggregate = None  # type: ignore[assignment]
    response_chunk_to_tts_aggregate: LatencyAggregate = None  # type: ignore[assignment]
    final_to_tts_aggregate: LatencyAggregate = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.audio_to_final_aggregate = LatencyAggregate()
        self.final_to_model_aggregate = LatencyAggregate()
        self.final_to_response_chunk_aggregate = LatencyAggregate()
        self.response_chunk_to_tts_aggregate = LatencyAggregate()
        self.final_to_tts_aggregate = LatencyAggregate()


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
            "final_to_first_model_token_ms": LatencyAggregate(),
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

    @staticmethod
    def _format_latency(latency_ms: float | None) -> str:
        return f"{latency_ms:.0f}" if latency_ms is not None else "n/a"

    def mark_call_started(
        self,
        conversation_id: str,
        *,
        from_phone: str | None,
        to_phone: str | None,
    ) -> None:
        now = perf_counter()
        with self._lock:
            state = self._get_state(conversation_id)
            state.started_at = now
            state.from_phone = from_phone
            state.to_phone = to_phone

        logger.info(
            "Latency tracking started for {}: from_phone={} to_phone={}",
            conversation_id,
            from_phone or "n/a",
            to_phone or "n/a",
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
            state.first_model_token_at = None
            state.first_llm_token_at = None
            state.first_tts_chunk_at = None
            state.last_transcript_preview = transcript[:80]

            audio_to_final_ms: float | None = None
            if state.last_audio_received_at is not None:
                audio_to_final_ms = (now - state.last_audio_received_at) * 1000
            if audio_to_final_ms is not None:
                state.audio_to_final_aggregate.record(audio_to_final_ms)
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

    def mark_first_model_token(self, conversation_id: str) -> None:
        now = perf_counter()
        with self._lock:
            state = self._get_state(conversation_id)
            if state.first_model_token_at is not None:
                return

            state.first_model_token_at = now

            final_to_model_ms: float | None = None
            if state.transcription_final_at is not None:
                final_to_model_ms = (now - state.transcription_final_at) * 1000
            if final_to_model_ms is not None:
                state.final_to_model_aggregate.record(final_to_model_ms)
            self._record_aggregate(
                "final_to_first_model_token_ms",
                final_to_model_ms,
                conversation_id=conversation_id,
                turn_index=state.turn_index,
            )

        logger.info(
            "Latency turn {} for {}: final_transcript->first_model_token={} ms",
            state.turn_index,
            conversation_id,
            f"{final_to_model_ms:.0f}" if final_to_model_ms is not None else "n/a",
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
            state.using_input_streaming_synthesizer = using_input_streaming_synthesizer

            final_to_llm_ms: float | None = None
            if state.transcription_final_at is not None:
                final_to_llm_ms = (now - state.transcription_final_at) * 1000
            if final_to_llm_ms is not None:
                state.final_to_response_chunk_aggregate.record(final_to_llm_ms)
            self._record_aggregate(
                "final_to_first_llm_ms",
                final_to_llm_ms,
                conversation_id=conversation_id,
                turn_index=state.turn_index,
            )

        logger.info(
            "Latency turn {} for {}: final_transcript->first_response_chunk={} ms input_streaming_synthesizer={}",
            state.turn_index,
            conversation_id,
            self._format_latency(final_to_llm_ms),
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
            final_to_model_ms: float | None = None
            audio_to_final_ms: float | None = None
            call_age_ms: float | None = None
            if state.first_llm_token_at is not None:
                llm_to_tts_ms = (now - state.first_llm_token_at) * 1000
            if state.transcription_final_at is not None:
                total_turn_ms = (now - state.transcription_final_at) * 1000
            if state.first_model_token_at is not None and state.transcription_final_at is not None:
                final_to_model_ms = (state.first_model_token_at - state.transcription_final_at) * 1000
            if state.last_audio_received_at is not None and state.transcription_final_at is not None:
                audio_to_final_ms = (state.transcription_final_at - state.last_audio_received_at) * 1000
            if state.started_at is not None:
                call_age_ms = (now - state.started_at) * 1000
            if llm_to_tts_ms is not None:
                state.response_chunk_to_tts_aggregate.record(llm_to_tts_ms)
            if total_turn_ms is not None:
                state.final_to_tts_aggregate.record(total_turn_ms)
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
            "Latency turn {} summary for {}: audio->final={} ms final->model={} ms final->first_response_chunk={} ms response_chunk->first_tts={} ms final->first_tts={} ms call_age={} ms input_streaming_synthesizer={} transcript={!r}",
            state.turn_index,
            conversation_id,
            self._format_latency(audio_to_final_ms),
            self._format_latency(final_to_model_ms),
            self._format_latency(
                (state.first_llm_token_at - state.transcription_final_at) * 1000
                if state.first_llm_token_at is not None and state.transcription_final_at is not None
                else None
            ),
            self._format_latency(llm_to_tts_ms),
            self._format_latency(total_turn_ms),
            self._format_latency(call_age_ms),
            state.using_input_streaming_synthesizer,
            state.last_transcript_preview or "",
        )

    def clear(self, conversation_id: str) -> None:
        with self._lock:
            state = self._states.pop(conversation_id, None)
        if state is None:
            return

        logger.info(
            "Latency call summary for {}: turns={} from_phone={} to_phone={} avg_audio->final={} ms avg_final->model={} ms avg_final->first_response_chunk={} ms avg_response_chunk->first_tts={} ms avg_final->first_tts={} ms",
            conversation_id,
            state.turn_index,
            state.from_phone or "n/a",
            state.to_phone or "n/a",
            self._format_latency(state.audio_to_final_aggregate.snapshot()["avg_ms"]),
            self._format_latency(state.final_to_model_aggregate.snapshot()["avg_ms"]),
            self._format_latency(state.final_to_response_chunk_aggregate.snapshot()["avg_ms"]),
            self._format_latency(state.response_chunk_to_tts_aggregate.snapshot()["avg_ms"]),
            self._format_latency(state.final_to_tts_aggregate.snapshot()["avg_ms"]),
        )

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "active_conversations": len(self._states),
                "segments": {
                    metric_name: aggregate.snapshot()
                    for metric_name, aggregate in self._aggregates.items()
                },
                "active_conversation_details": {
                    conversation_id: {
                        "turn_index": state.turn_index,
                        "from_phone": state.from_phone,
                        "to_phone": state.to_phone,
                        "last_transcript_preview": state.last_transcript_preview,
                        "input_streaming_synthesizer": state.using_input_streaming_synthesizer,
                    }
                    for conversation_id, state in self._states.items()
                },
                "recent_turns": list(self._recent_turns),
            }


conversation_latency_tracker = ConversationLatencyTracker()
