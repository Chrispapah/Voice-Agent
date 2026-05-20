"""Buffer streamed LLM tokens and emit whole sentences for TTS (browser voice).

Sentence end: ``.`` ``!`` ``?`` (runs like ``...`` allowed), Unicode Greek question mark
(U+037E), optional closing quotes/brackets, then whitespace. Remainder is flushed on ``flush()``.
If the buffer grows past ``max_buffer_chars`` without a sentence break, flush at the last
space in the tail window (or the whole buffer if there is no space) so streaming cannot stall.
"""

from __future__ import annotations

# Greek question mark (erotimatiko); distinct from ASCII semicolon U+003B.
_GREEK_QUESTION_MARK = "\u037e"


def find_first_sentence_end(s: str) -> int | None:
    """Return exclusive index after trailing whitespace following the first sentence end.

    ``None`` if no complete sentence is present yet (caller may still need overflow handling).
    """
    n = len(s)
    i = 0
    while i < n:
        c = s[i]
        if c in "!?" or c == _GREEK_QUESTION_MARK:
            j = i + 1
            while j < n and s[j] in "!?":
                j += 1
            j = _skip_closing_glyphs(s, j, n)
            if j < n and s[j].isspace():
                return _consume_trailing_ws(s, j, n)
            i = j
            continue
        if c == ".":
            j = i + 1
            while j < n and s[j] == ".":
                j += 1
            j = _skip_closing_glyphs(s, j, n)
            if j < n and s[j].isspace():
                return _consume_trailing_ws(s, j, n)
            i = j
            continue
        i += 1
    return None


def _skip_closing_glyphs(s: str, j: int, n: int) -> int:
    while j < n and s[j] in ")]'\"»\u201d\u2019":
        j += 1
    return j


def _consume_trailing_ws(s: str, j: int, n: int) -> int:
    k = j
    while k < n and s[k].isspace():
        k += 1
    return k


def overflow_split_index(s: str, *, tail_window: int = 240) -> int:
    """Break an over-long buffer without a sentence end; prefer last space in the tail."""
    n = len(s)
    start = max(0, n - tail_window)
    sp = s.rfind(" ", start, n)
    if sp > 0:
        return sp + 1
    sp2 = s.rfind(" ", 0, n)
    if sp2 > 0:
        return sp2 + 1
    return n


class SentenceStreamBuffer:
    """Accumulates LLM token chunks and emits sentence-sized strings for TTS."""

    def __init__(self, *, max_buffer_chars: int = 800) -> None:
        self.buffer = ""
        self._max = max_buffer_chars

    async def feed(self, on_sentence, chunk: str) -> None:
        if not chunk:
            return
        self.buffer += chunk
        while self.buffer:
            end = find_first_sentence_end(self.buffer)
            if end is not None:
                await self._emit_prefix(on_sentence, end)
                continue
            if len(self.buffer) >= self._max:
                split_at = overflow_split_index(self.buffer)
                if split_at <= 0:
                    break
                await self._emit_prefix(on_sentence, split_at)
                continue
            break

    async def flush(self, on_sentence) -> None:
        if not self.buffer:
            return
        buffered = self.buffer
        self.buffer = ""
        text = buffered if buffered.endswith(" ") else buffered + " "
        await on_sentence(text)

    async def _emit_prefix(self, on_sentence, end: int) -> None:
        output = self.buffer[:end].strip()
        self.buffer = self.buffer[end:].lstrip()
        if output:
            text = output if output.endswith(" ") else output + " "
            await on_sentence(text)
