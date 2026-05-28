from __future__ import annotations

import re

_ENUM_PATTERN = re.compile(r"SamplingRate\.RATE_(\d+)")


def resolve_web_voice_deepgram_model(raw_model: str | None) -> str:
    """Browser voice sends MediaRecorder WebM/Opus, not 8 kHz mulaw.

    The telephony ``phonecall`` model targets phone audio; using it for browser streams with
    some locales can make Deepgram reject the WebSocket with HTTP 400.
    Map those to a general Nova model."""
    model = (raw_model or "").strip() or "nova-2"
    lower = model.lower()
    if lower == "phonecall" or "phonecall" in lower:
        return "nova-2"
    return model


def normalize_deepgram_language_code(code: str | None) -> str:
    """Map BCP-47 style codes to Deepgram live-stream language tokens where needed."""
    raw = (code or "el").strip()
    norm = raw.lower().replace("_", "-")
    if norm == "el" or norm.startswith("el-"):
        return "el"
    return raw


def prefer_nova3_for_greek_browser_stt(model: str, language_code: str | None) -> str:
    """Nova 3 reports stronger Greek support than Nova 2 for browser (non-phonecall) audio."""
    if normalize_deepgram_language_code(language_code) != "el":
        return model
    m = (model or "").strip().lower()
    if m == "nova-2":
        return "nova-3"
    if m.startswith("nova-2-") and m != "nova-2-phonecall":
        return "nova-3"
    return model


def patch_deepgram_url_enums(url: str) -> str:
    """Fix enum serialization bugs in legacy Deepgram URL builders."""
    return _ENUM_PATTERN.sub(r"\1", url)
