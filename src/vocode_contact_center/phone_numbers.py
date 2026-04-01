from __future__ import annotations

import re

import phonenumbers
from phonenumbers import NumberParseException, PhoneNumberFormat


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


def _clean_phone_number(raw_phone_number: str) -> str:
    candidate = raw_phone_number.strip()
    if not candidate:
        return ""

    if candidate.startswith("00"):
        candidate = f"+{candidate[2:]}"

    digits = re.sub(r"\D", "", candidate)
    if candidate.startswith("+"):
        return f"+{digits}"
    return digits
