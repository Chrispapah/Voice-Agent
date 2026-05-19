from ai_sdr_agent.text.greek_number_words import (
    expand_digit_runs_for_greek_tts,
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
