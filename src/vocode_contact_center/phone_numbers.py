from __future__ import annotations

import functools
import re

import phonenumbers
from phonenumbers import NumberParseException, PhoneNumberFormat

_DIGIT_WORDS = {
    "zero": "0",
    "oh": "0",
    "o": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}

_TEEN_WORDS = {
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
}

_TENS_WORDS = {
    "twenty": "20",
    "thirty": "30",
    "forty": "40",
    "fifty": "50",
    "sixty": "60",
    "seventy": "70",
    "eighty": "80",
    "ninety": "90",
}


def parse_spoken_digit_sequence(raw_text: str) -> str:
    return _parse_spoken_numeric_sequence(raw_text, allow_plus=False)


def extract_caller_e164_from_call_context(call_context: str) -> str | None:
    """Best-effort E.164 from telephony call context (e.g. sip:+30698...@host in Caller number line)."""
    if not call_context or not call_context.strip():
        return None
    sip_match = re.search(r"sip:(\+[1-9]\d{6,14})@", call_context, re.IGNORECASE)
    if sip_match:
        candidate = sip_match.group(1)
        return _validate_and_format_e164(candidate)
    loose = re.search(
        r"(?:caller\s+number|from)\s*:\s*(\+[1-9]\d{6,14})\b",
        call_context,
        re.IGNORECASE,
    )
    if loose:
        return _validate_and_format_e164(loose.group(1))
    return None


def _validate_and_format_e164(candidate: str) -> str | None:
    try:
        parsed = phonenumbers.parse(candidate, None)
    except NumberParseException:
        return None
    if not phonenumbers.is_valid_number(parsed):
        return None
    return phonenumbers.format_number(parsed, PhoneNumberFormat.E164)


def normalize_phone_number(
    raw_phone_number: str,
    *,
    default_region: str | None = None,
) -> str | None:
    cleaned = _clean_phone_number(raw_phone_number, default_region=default_region)
    if not cleaned:
        return None

    region = default_region.strip().upper() if default_region else None
    try:
        parsed = phonenumbers.parse(cleaned, region)
    except NumberParseException:
        return None

    if not phonenumbers.is_possible_number(parsed):
        return None
    if not phonenumbers.is_valid_number(parsed):
        return None
    return phonenumbers.format_number(parsed, PhoneNumberFormat.E164)


@functools.lru_cache(maxsize=256)
def normalize_phone_number_cached(
    raw_phone_number: str,
    default_region: str | None,
) -> str | None:
    """LRU-cached E.164 normalization for repeated inputs (e.g. auth re-prompts)."""
    return normalize_phone_number(raw_phone_number, default_region=default_region)


def _try_clean_country_code_phone_number_clause(
    raw_phone_number: str,
    default_region: str | None,
) -> str | None:
    """Handle 'country code … phone number …' spoken form without merging both into one digit string."""
    lowered = raw_phone_number.lower()
    cc_marker = "country code"
    pn_marker = "phone number"
    if cc_marker not in lowered or pn_marker not in lowered:
        return None
    i_cc = lowered.index(cc_marker)
    i_pn = lowered.index(pn_marker)
    if i_pn <= i_cc:
        return None
    country_segment = raw_phone_number[i_cc + len(cc_marker) : i_pn].strip(" ,.:;")
    national_segment = raw_phone_number[i_pn + len(pn_marker) :].strip(" ,.:;")
    if not country_segment or not national_segment:
        return None
    cc_digits = _parse_country_calling_code_spoken(country_segment, default_region)
    if not cc_digits:
        return None
    national_digits = _parse_spoken_numeric_sequence(national_segment, allow_plus=False)
    if not national_digits:
        return None
    return f"+{cc_digits}{national_digits}"


def _parse_country_calling_code_spoken(segment: str, default_region: str | None) -> str | None:
    raw = _parse_spoken_numeric_sequence(segment, allow_plus=True)
    if not raw:
        return None
    cc = raw[1:] if raw.startswith("+") else raw
    region = default_region.strip().upper() if default_region else None
    if region == "GR" and cc == "13":
        cc = "30"
    if not cc.isdigit() or not (1 <= len(cc) <= 3):
        return None
    return cc


def _clean_phone_number(raw_phone_number: str, default_region: str | None = None) -> str:
    candidate = raw_phone_number.strip()
    if not candidate:
        return ""

    split_cleaned = _try_clean_country_code_phone_number_clause(candidate, default_region)
    if split_cleaned:
        return split_cleaned

    explicit_country_code = _extract_spoken_country_code_suffix(candidate)
    if explicit_country_code is not None:
        local_part = candidate.lower().split("country code", 1)[0]
        local_number = _parse_spoken_numeric_sequence(local_part, allow_plus=False)
        if local_number:
            return f"{explicit_country_code}{local_number}"

    candidate = candidate.replace("plus", "+")
    if candidate.startswith("00"):
        candidate = f"+{candidate[2:]}"

    digits = re.sub(r"\D", "", candidate)
    if not digits:
        spoken_candidate = _parse_spoken_phone_number(candidate)
        if not spoken_candidate:
            return ""
        candidate = spoken_candidate
        digits = re.sub(r"\D", "", candidate)

    if candidate.startswith("+"):
        return f"+{digits}"
    return digits


def _parse_spoken_phone_number(raw_phone_number: str) -> str:
    return _parse_spoken_numeric_sequence(raw_phone_number, allow_plus=True)


def _extract_spoken_country_code_suffix(raw_text: str) -> str | None:
    lowered = raw_text.lower()
    marker = "country code"
    if marker not in lowered:
        return None
    if "phone number" in lowered and lowered.index("phone number") > lowered.index(marker):
        return None
    suffix = lowered.split(marker, 1)[1].strip(" ,.:;")
    if not suffix:
        return None
    parsed_suffix = _parse_spoken_numeric_sequence(suffix, allow_plus=True)
    if parsed_suffix.startswith("+") and len(parsed_suffix) > 1:
        return parsed_suffix
    return None


def _parse_spoken_numeric_sequence(raw_text: str, *, allow_plus: bool) -> str:
    tokens = re.findall(r"[a-zA-Z]+|\+", raw_text.lower())
    if not tokens:
        return ""

    pieces: list[str] = []
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if token == "+":
            if allow_plus and not pieces:
                pieces.append("+")
            idx += 1
            continue
        if token in {"plus", "pass"}:
            if allow_plus and not pieces:
                pieces.append("+")
            idx += 1
            continue
        if token in {"double", "triple"}:
            repeat = 2 if token == "double" else 3
            if idx + 1 < len(tokens) and tokens[idx + 1] in _DIGIT_WORDS:
                pieces.append(_DIGIT_WORDS[tokens[idx + 1]] * repeat)
                idx += 2
                continue
            idx += 1
            continue
        if token in _DIGIT_WORDS:
            pieces.append(_DIGIT_WORDS[token])
            idx += 1
            continue
        if token in _TEEN_WORDS:
            pieces.append(_TEEN_WORDS[token])
            idx += 1
            continue
        if token in _TENS_WORDS:
            combined = _TENS_WORDS[token]
            if idx + 1 < len(tokens) and tokens[idx + 1] in _DIGIT_WORDS:
                combined = str(int(combined) + int(_DIGIT_WORDS[tokens[idx + 1]]))
                idx += 1
            pieces.append(combined)
            idx += 1
            continue
        idx += 1

    return "".join(pieces)
