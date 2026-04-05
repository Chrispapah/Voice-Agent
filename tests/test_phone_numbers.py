from vocode_contact_center.phone_numbers import normalize_phone_number, parse_spoken_digit_sequence


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
