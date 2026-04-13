"""In-process turn latency samples for ops and tuning (p50/p95, by route)."""

from __future__ import annotations

import asyncio
import statistics
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

from loguru import logger


@dataclass(frozen=True)
class TurnLatencySample:
    conversation_id: str
    turn_count: int
    route_decision: str
    latency_total_ms: float
    latency_graph_ms: float
    latency_persist_ms: float
    recorded_at: float


@dataclass(frozen=True)
class PerceivedTurnSample:
    """Phone turn timings in one worker (perf_counter-based).

    stt_final_* are set when Deepgram enqueues a final transcript (see transcriber hook).
    """

    conversation_id: str
    graph_ms: float
    post_graph_to_first_audio_ms: float
    perceived_total_ms: float
    stt_final_to_respond_ms: float | None
    stt_final_to_first_audio_ms: float | None
    recorded_at: float


_turn_perf: dict[str, tuple[float, float | None, float | None]] = {}
"""conversation_id -> (respond_enter_pc, graph_done_pc | None, stt_final_emit_pc | None)"""

_turn_lock = threading.Lock()
# Latest Deepgram final transcript enqueue time per call (consumption in respond_enter).
_pending_stt_final_pc: dict[str, float] = {}


def mark_deepgram_final_transcript_enqueued_from_context() -> None:
    """Call when Deepgram puts a final Transcription on the transcriber output queue (telephony)."""
    try:
        from vocode import conversation_id as vocode_conversation_id

        cid = vocode_conversation_id.value
    except (LookupError, RuntimeError):
        return
    if not cid or not isinstance(cid, str):
        return
    with _turn_lock:
        _pending_stt_final_pc[cid] = time.perf_counter()


def mark_phone_turn_respond_enter(conversation_id: str) -> None:
    """Call at start of SDRVocodeAgent.respond (final text handed to agent)."""
    with _turn_lock:
        t_stt = _pending_stt_final_pc.pop(conversation_id, None)
        t0 = time.perf_counter()
        _turn_perf[conversation_id] = (t0, None, t_stt)


def mark_phone_turn_graph_done(conversation_id: str) -> None:
    """Call after graph/brain work completes, before returning text to vocode for TTS."""
    with _turn_lock:
        cur = _turn_perf.get(conversation_id)
        if cur is None:
            return
        t0, _, t_stt = cur
        _turn_perf[conversation_id] = (t0, time.perf_counter(), t_stt)


def clear_phone_turn_on_error(conversation_id: str) -> None:
    with _turn_lock:
        _turn_perf.pop(conversation_id, None)


def _finalize_perceived_turn(conversation_id: str, t_audio: float) -> PerceivedTurnSample | None:
    with _turn_lock:
        cur = _turn_perf.pop(conversation_id, None)
    if cur is None:
        return None
    t0, t1, t_stt = cur
    if t1 is None:
        t1 = t0
    graph_ms = (t1 - t0) * 1000.0
    post_ms = (t_audio - t1) * 1000.0
    total_ms = (t_audio - t0) * 1000.0
    stt_to_resp = (t0 - t_stt) * 1000.0 if t_stt is not None else None
    stt_to_audio = (t_audio - t_stt) * 1000.0 if t_stt is not None else None
    return PerceivedTurnSample(
        conversation_id=conversation_id,
        graph_ms=graph_ms,
        post_graph_to_first_audio_ms=post_ms,
        perceived_total_ms=total_ms,
        stt_final_to_respond_ms=stt_to_resp,
        stt_final_to_first_audio_ms=stt_to_audio,
        recorded_at=time.time(),
    )


def note_first_tts_audio_chunk_from_context() -> None:
    """Call when the first non-empty audio chunk is produced (ElevenLabs HTTP/WS)."""
    try:
        from vocode import conversation_id as vocode_conversation_id

        cid = vocode_conversation_id.value
    except (LookupError, RuntimeError):
        return
    if not cid or not isinstance(cid, str):
        return
    sample = _finalize_perceived_turn(cid, time.perf_counter())
    if sample is None:
        return
    logger.info(
        "perceived_latency conversation_id={} stt_final_to_respond_ms={} stt_final_to_first_audio_ms={} "
        "graph_ms={:.0f} post_graph_to_first_audio_ms={:.0f} respond_to_first_audio_ms={:.0f}",
        sample.conversation_id,
        f"{sample.stt_final_to_respond_ms:.0f}"
        if sample.stt_final_to_respond_ms is not None
        else "n/a",
        f"{sample.stt_final_to_first_audio_ms:.0f}"
        if sample.stt_final_to_first_audio_ms is not None
        else "n/a",
        sample.graph_ms,
        sample.post_graph_to_first_audio_ms,
        sample.perceived_total_ms,
    )
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    loop.create_task(shared_latency_analytics.record_perceived_turn(sample))


def _percentile_sorted(sorted_vals: list[float], p: float) -> float | None:
    if not sorted_vals:
        return None
    n = len(sorted_vals)
    idx = min(n - 1, max(0, int(round((p / 100.0) * (n - 1)))))
    return sorted_vals[idx]


def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "count": 0,
            "min_ms": None,
            "max_ms": None,
            "mean_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "p99_ms": None,
        }
    s = sorted(values)
    return {
        "count": len(s),
        "min_ms": s[0],
        "max_ms": s[-1],
        "mean_ms": statistics.fmean(s),
        "p50_ms": _percentile_sorted(s, 50),
        "p95_ms": _percentile_sorted(s, 95),
        "p99_ms": _percentile_sorted(s, 99),
    }


class LatencyAnalyticsBuffer:
    """Thread-safe (asyncio lock) ring buffer of per-turn latencies."""

    def __init__(self, maxlen: int = 10_000):
        self._maxlen = maxlen
        self._samples: deque[TurnLatencySample] = deque(maxlen=maxlen)
        self._perceived: deque[PerceivedTurnSample] = deque(maxlen=maxlen)
        self._lock = asyncio.Lock()

    async def record_perceived_turn(self, sample: PerceivedTurnSample) -> None:
        async with self._lock:
            self._perceived.append(sample)

    async def record_turn(
        self,
        *,
        conversation_id: str,
        turn_count: int,
        route_decision: str,
        latency_total_ms: float,
        latency_graph_ms: float,
        latency_persist_ms: float,
    ) -> None:
        sample = TurnLatencySample(
            conversation_id=conversation_id,
            turn_count=turn_count,
            route_decision=route_decision,
            latency_total_ms=latency_total_ms,
            latency_graph_ms=latency_graph_ms,
            latency_persist_ms=latency_persist_ms,
            recorded_at=time.time(),
        )
        async with self._lock:
            self._samples.append(sample)

    async def snapshot(self, *, recent_limit: int = 50) -> dict[str, Any]:
        async with self._lock:
            items = list(self._samples)
            perceived_items = list(self._perceived)
        total = [x.latency_total_ms for x in items]
        graph = [x.latency_graph_ms for x in items]
        persist = [x.latency_persist_ms for x in items]
        by_route: dict[str, list[float]] = {}
        for x in items:
            by_route.setdefault(x.route_decision, []).append(x.latency_total_ms)

        recent = [asdict(x) for x in items[-recent_limit:]]
        p_tot = [x.perceived_total_ms for x in perceived_items]
        p_post = [x.post_graph_to_first_audio_ms for x in perceived_items]
        p_graph = [x.graph_ms for x in perceived_items]
        stt_resp = [x.stt_final_to_respond_ms for x in perceived_items if x.stt_final_to_respond_ms is not None]
        stt_audio = [
            x.stt_final_to_first_audio_ms
            for x in perceived_items
            if x.stt_final_to_first_audio_ms is not None
        ]
        recent_perceived = [asdict(x) for x in perceived_items[-recent_limit:]]
        return {
            "buffer_maxlen": self._maxlen,
            "sample_count": len(items),
            "latency_total_ms": _stats(total),
            "latency_graph_ms": _stats(graph),
            "latency_persist_ms": _stats(persist),
            "by_route_decision": {
                route: _stats(vals) for route, vals in sorted(by_route.items())
            },
            "recent_turns": recent,
            "perceived_phone": {
                "sample_count": len(perceived_items),
                "respond_to_first_audio_ms": _stats(p_tot),
                "graph_ms": _stats(p_graph),
                "post_graph_to_first_audio_ms": _stats(p_post),
                "stt_final_to_respond_ms": _stats(stt_resp),
                "stt_final_to_first_audio_ms": _stats(stt_audio),
                "recent": recent_perceived,
            },
        }


# One buffer per process (Railway / uvicorn worker). Multi-worker = per-replica stats.
shared_latency_analytics = LatencyAnalyticsBuffer()
