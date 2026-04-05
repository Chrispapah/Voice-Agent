from vocode_contact_center.phone_numbers import (
    extract_caller_e164_from_call_context,
    normalize_phone_number,
    parse_spoken_digit_sequence,
)


def test_normalize_phone_number_accepts_spoken_digits_with_default_region():
    assert (
        normalize_phone_number(
            "Six nine eighty eight zero three ninety nine seventy one",
            default_region="GR",
        )
        == "+306988039971"
    )


def test_normalize_phone_number_accepts_spoken_country_code():
    assert (
        normalize_phone_number(
            "plus thirty zero six nine eighty eight zero three ninety nine seventy one",
        )
        == "+306988039971"
    )


def test_normalize_phone_number_accepts_us_spoken_digits():
    assert (
        normalize_phone_number(
            "four one five five five five two six seven one",
            default_region="US",
        )
        == "+14155552671"
    )


def test_parse_spoken_digit_sequence_accepts_confirmation_code_words():
    assert parse_spoken_digit_sequence("Nine four seven one nine nine") == "947199"


def test_normalize_phone_number_accepts_spoken_local_number_then_country_code_suffix():
    assert (
        normalize_phone_number(
            "Six nine eight eight zero three nine nine seven one. Country code, plus thirty",
            default_region="GR",
        )
        == "+306988039971"
    )


def test_normalize_phone_number_accepts_country_code_then_phone_number_clause():
    assert (
        normalize_phone_number(
            "Country code, plus thirty. Phone number, six nine eight eight zero three nine nine seven one.",
            default_region="GR",
        )
        == "+306988039971"
    )


def test_normalize_phone_number_accepts_pass_homophone_as_plus_in_country_clause():
    assert (
        normalize_phone_number(
            "Country code, pass thirty. Phone number, six nine eight eight zero three nine nine seven one.",
            default_region="GR",
        )
        == "+306988039971"
    )


def test_normalize_phone_number_gr_maps_misheard_thirteen_to_thirty_in_country_clause():
    assert (
        normalize_phone_number(
            "Country code, plus thirteen. Phone number, six nine eight eight zero three nine nine seven one.",
            default_region="GR",
        )
        == "+306988039971"
    )


def test_normalize_phone_number_thirteen_country_code_unchanged_without_gr_region():
    assert (
        normalize_phone_number(
            "Country code, plus thirteen. Phone number, six nine eight eight zero three nine nine seven one.",
            default_region="US",
        )
        is None
    )


def test_normalize_phone_number_plus_thirty_six_still_invalid_without_extra_heuristic():
    assert (
        normalize_phone_number(
            "Plus thirty six nine eight eight zero three nine nine seven one.",
            default_region="GR",
        )
        is None
    )


def test_extract_caller_e164_from_sip_in_call_context():
    ctx = "Live call metadata:\n- Caller number: sip:+306988039971@pbx.example.com\n"
    assert extract_caller_e164_from_call_context(ctx) == "+306988039971"


def test_extract_caller_e164_returns_none_when_missing():
    assert extract_caller_e164_from_call_context("Live call metadata:\n- Caller metadata was not available\n") is None
