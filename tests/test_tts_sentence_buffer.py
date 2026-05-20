import asyncio

import pytest

from ai_sdr_agent.text.tts_sentence_buffer import (
    SentenceStreamBuffer,
    find_first_sentence_end,
    overflow_split_index,
)


def test_find_first_sentence_end_basic():
    assert find_first_sentence_end("Hello. There") == 7
    assert find_first_sentence_end("Hello.  There") == 8


def test_find_first_sentence_end_question_exclamation():
    assert find_first_sentence_end("Really? Yes") == 8
    assert find_first_sentence_end("Go! Now") == 4


def test_find_first_sentence_end_greek_question_mark():
    s = f"Πώς είσαι{chr(0x037E)} Καλά"
    end = find_first_sentence_end(s)
    assert end is not None
    assert s[:end].strip() == f"Πώς είσαι{chr(0x037E)}"


def test_find_first_sentence_end_ellipsis():
    assert find_first_sentence_end("Wait... OK") == 8


def test_find_first_sentence_end_incomplete():
    assert find_first_sentence_end("Hello") is None
    assert find_first_sentence_end("Hello.") is None


def test_overflow_split_index():
    s = "a" * 100 + " " + "b" * 100
    idx = overflow_split_index(s, tail_window=50)
    assert idx == 101


def test_sentence_stream_buffer_emits_per_sentence():
    out: list[str] = []

    async def _run() -> None:
        async def on_sentence(t: str) -> None:
            out.append(t.strip())

        buf = SentenceStreamBuffer(max_buffer_chars=10_000)
        await buf.feed(on_sentence, "First bit. ")
        await buf.feed(on_sentence, "Second bit? ")
        await buf.feed(on_sentence, "Third")
        assert out == ["First bit.", "Second bit?"]
        await buf.flush(on_sentence)
        assert out == ["First bit.", "Second bit?", "Third"]

    asyncio.run(_run())


def test_sentence_stream_buffer_overflow_without_punct():
    out: list[str] = []

    async def _run() -> None:
        async def on_sentence(t: str) -> None:
            out.append(t.strip())

        buf = SentenceStreamBuffer(max_buffer_chars=20)
        chunk = "word " * 10  # long, no period
        await buf.feed(on_sentence, chunk)
        assert len(out) >= 1
        assert all(len(x) > 0 for x in out)

    asyncio.run(_run())
