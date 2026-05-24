from __future__ import annotations

import re
import time
from dataclasses import dataclass
from difflib import SequenceMatcher


_NON_WORD_RE = re.compile(r"[^\w\s]+", re.UNICODE)
_SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class EchoMatch:
    transcript: str
    agent_text: str
    score: float
    reason: str


def normalize_for_echo_match(text: str) -> str:
    lowered = text.casefold().strip()
    without_punctuation = _NON_WORD_RE.sub(" ", lowered)
    return _SPACE_RE.sub(" ", without_punctuation).strip()


class RealtimeEchoGuard:
    """Drop transcripts that look like recent agent playback leaking into the mic."""

    def __init__(
        self,
        *,
        similarity_threshold: float = 0.72,
        min_window_s: float = 2.5,
        max_window_s: float = 14.0,
        hangover_s: float = 1.5,
        chars_per_second: float = 14.0,
    ) -> None:
        self._similarity_threshold = similarity_threshold
        self._min_window_s = min_window_s
        self._max_window_s = max_window_s
        self._hangover_s = hangover_s
        self._chars_per_second = chars_per_second
        self._candidates: list[tuple[str, str, float]] = []

    def record_agent_speech(self, text: str, *, now: float | None = None) -> None:
        normalized = normalize_for_echo_match(text)
        if not normalized:
            return
        recorded_at = time.perf_counter() if now is None else now
        estimated_speech_s = len(normalized) / self._chars_per_second
        window_s = max(self._min_window_s, min(self._max_window_s, estimated_speech_s + self._hangover_s))
        self._purge(recorded_at)
        self._candidates.append((normalized, text.strip(), recorded_at + window_s))

    def check(self, transcript: str, *, now: float | None = None) -> EchoMatch | None:
        normalized_transcript = normalize_for_echo_match(transcript)
        if not normalized_transcript:
            return None
        checked_at = time.perf_counter() if now is None else now
        self._purge(checked_at)

        best: EchoMatch | None = None
        for normalized_agent_text, raw_agent_text, _expires_at in self._candidates:
            match = self._match_candidate(
                transcript=transcript,
                normalized_transcript=normalized_transcript,
                raw_agent_text=raw_agent_text,
                normalized_agent_text=normalized_agent_text,
            )
            if match and (best is None or match.score > best.score):
                best = match
        return best

    def _match_candidate(
        self,
        *,
        transcript: str,
        normalized_transcript: str,
        raw_agent_text: str,
        normalized_agent_text: str,
    ) -> EchoMatch | None:
        score = SequenceMatcher(None, normalized_transcript, normalized_agent_text).ratio()
        if score >= self._similarity_threshold:
            return EchoMatch(
                transcript=transcript,
                agent_text=raw_agent_text,
                score=score,
                reason="similarity",
            )
        return None

    def _purge(self, now: float) -> None:
        self._candidates = [candidate for candidate in self._candidates if candidate[2] >= now]
