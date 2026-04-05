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


def normalize_phone_number(
    raw_phone_number: str,
    *,
    default_region: str | None = None,
) -> str | None:
    cleaned = _clean_phone_number(raw_phone_number)
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


def _clean_phone_number(raw_phone_number: str) -> str:
    candidate = raw_phone_number.strip()
    if not candidate:
        return ""

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
        if token in {"plus"}:
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
