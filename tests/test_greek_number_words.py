from ai_sdr_agent.text.greek_number_words import (
    expand_digit_runs_for_greek_tts,
    expand_for_greek_elevenlabs_tts,
    integer_to_greek_cardinal_words,
)


def test_expand_digit_run_to_el_words():
    out = expand_digit_runs_for_greek_tts("Η τιμή είναι 25.")
    assert "25" not in out
    assert "είκοσι" in out
    assert "πέντε" in out


def test_thousands_block():
    assert "χίλια" in integer_to_greek_cardinal_words(1525)
    assert "πεντακόσια" in integer_to_greek_cardinal_words(1525)


def test_preserves_long_digit_sequences():
    raw = "+3069412345678"
    assert expand_digit_runs_for_greek_tts(raw) == raw


def test_preserves_leading_zero_runs():
    raw = "Κώδικας 0123"
    assert expand_digit_runs_for_greek_tts(raw) == raw


def test_expand_com_and_gr_for_elevenlabs():
    out = expand_for_greek_elevenlabs_tts("Δες example.COM και x.GR τέλος")
    assert ".com" not in out.lower()
    assert ".gr" not in out.lower()
    assert "τελεία κομ" in out
    assert "τελεία τζι αρ" in out


def test_com_gr_order_double_tld():
    """`.com` is rewritten before `.gr`; both should appear as spoken chunks."""
    out = expand_for_greek_elevenlabs_tts("shop.example.com.gr")
    assert "τελεία κομ" in out and "τελεία τζι αρ" in out
    assert ".com" not in out and ".gr" not in out
