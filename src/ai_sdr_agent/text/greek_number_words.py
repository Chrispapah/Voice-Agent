"""Expand Arabic numerals to Greek cardinal phrases before ElevenLabs TTS.

``num2words`` does not implement Greek (`lang='el'` raises NotImplementedError), so this
module uses a small handwritten cardinal formatter suitable for conversational SDR prompts.
"""

from __future__ import annotations

import re

_UNITS_NEUTRAL = ["μηδέν", "ένα", "δύο", "τρία", "τέσσερα", "πέντε", "έξι", "επτά", "οκτώ", "εννέα"]

_TEENS = {
    10: "δέκα",
    11: "έντεκα",
    12: "δώδεκα",
    13: "δεκατρία",
    14: "δεκατέσσερα",
    15: "δεκαπέντε",
    16: "δεκαέξι",
    17: "δεκαεπτά",
    18: "δεκαοκτώ",
    19: "δεκαεννέα",
}

_TENS_UNIT = ["", "", "είκοσι", "τριάντα", "σαράντα", "πενήντα", "εξήντα", "εβδομήντα", "ογδόντα", "ενενήντα"]

_HUNDRED_TO_900_NEUTRAL = [
    "",
    "εκατό",
    "διακόσια",
    "τριακόσια",
    "τετρακόσια",
    "πεντακόσια",
    "εξακόσια",
    "επτακόσια",
    "οκτακόσια",
    "εννιακόσια",
]


def _under_99(n: int) -> str:
    if n < 0 or n >= 100:
        msg = "internal: _under_99 expects 0..99"
        raise ValueError(msg)
    if n < 10:
        return _UNITS_NEUTRAL[n]
    if n < 20:
        return _TEENS[n]
    ten, unit = divmod(n, 10)
    ten_w = _TENS_UNIT[ten]
    return f"{ten_w} {_UNITS_NEUTRAL[unit]}".strip() if unit else ten_w


def _under_999(n: int) -> str:
    if n < 0 or n >= 1000:
        msg = "internal: _under_999 expects 0..999"
        raise ValueError(msg)
    if n == 0:
        return ""
    h, remainder = divmod(n, 100)
    parts: list[str] = []

    if h == 1:
        if remainder == 0:
            parts.append("εκατό")
        else:
            parts.append(f"εκατόν {_under_99(remainder)}".strip())
    elif h >= 2:
        stem = _HUNDRED_TO_900_NEUTRAL[h]
        if remainder:
            parts.append(f"{stem} {_under_99(remainder)}".strip())
        else:
            parts.append(stem)
    elif remainder > 0:
        parts.append(_under_99(remainder))

    return " ".join(parts)


def integer_to_greek_cardinal_words(n: int) -> str:
    """Γράφει ακεραίους 0 έως 999_999 σε απλά καρδιναλ ουδετέρα (ΤTS)."""
    if not 0 <= n <= 999_999:
        raise ValueError("expected integer 0 <= n <= 999_999")

    rest = n
    if rest == 0:
        return "μηδέν"
    thousand_block, remainder = divmod(rest, 1000)
    parts: list[str] = []

    if thousand_block == 1:
        if remainder == 0:
            parts.append("χίλια")
        else:
            parts.append(f"χίλια {_under_999(remainder)}".strip())
    elif thousand_block >= 2:
        high = _under_999(thousand_block).strip()
        if remainder == 0:
            parts.append(f"{high} χιλιάδες")
        else:
            parts.append(f"{high} χιλιάδες {_under_999(remainder)}".strip())
    else:
        core = _under_999(rest)
        parts.append(core)

    return " ".join(p for p in parts if p).strip()


# Runs longer than this are treated as identifiers / phones and left as digits.
_MAX_DIGITS_IN_RUN = 7

_digit_run_re = re.compile(r"\d+")


def expand_digit_runs_for_greek_tts(text: str) -> str:
    """Replace digit runs with Greek cardinal phrases.

    Leaves long sequences and digit runs whose first symbol is ``0`` (e.g. local codes)
    untouched so callers can still pronounce phone blobs as digits when desired.
    """
    if not text:
        return text

    def replace_run(m: re.Match[str]) -> str:
        raw = m.group(0)
        if len(raw) > _MAX_DIGITS_IN_RUN:
            return raw
        if len(raw) > 1 and raw.startswith("0"):
            return raw
        try:
            n = int(raw)
            if n > 999_999:
                return raw
        except ValueError:
            return raw
        try:
            return integer_to_greek_cardinal_words(n)
        except ValueError:
            return raw

    return _digit_run_re.sub(replace_run, text)
