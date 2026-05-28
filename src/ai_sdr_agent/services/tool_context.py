from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

ToolSoundCallback = Callable[[dict[str, Any]], Awaitable[None]]


@dataclass
class ActiveToolVoiceFlags:
    disable_interruptions: bool = False
    pre_tool_speech: str = "auto"
    pre_tool_speech_text: str | None = None
    tool_call_sound: str = "none"
    tool_call_sound_url: str | None = None


_ACTIVE_TOOL_VOICE: ContextVar[ActiveToolVoiceFlags | None] = ContextVar(
    "active_tool_voice_flags",
    default=None,
)
_TOOL_SOUND_CALLBACK: ContextVar[ToolSoundCallback | None] = ContextVar(
    "tool_sound_callback",
    default=None,
)


def set_active_tool_voice(flags: ActiveToolVoiceFlags) -> Token:
    return _ACTIVE_TOOL_VOICE.set(flags)


def reset_active_tool_voice(token: Token) -> None:
    _ACTIVE_TOOL_VOICE.reset(token)


def get_active_tool_voice() -> ActiveToolVoiceFlags | None:
    return _ACTIVE_TOOL_VOICE.get()


def tool_interruptions_disabled() -> bool:
    flags = get_active_tool_voice()
    return bool(flags and flags.disable_interruptions)


def set_tool_sound_callback(cb: ToolSoundCallback | None) -> Token:
    return _TOOL_SOUND_CALLBACK.set(cb)


def reset_tool_sound_callback(token: Token) -> None:
    _TOOL_SOUND_CALLBACK.reset(token)


async def emit_tool_sound(payload: dict[str, Any]) -> None:
    cb = _TOOL_SOUND_CALLBACK.get()
    if cb is not None:
        await cb(payload)


def voice_interruptions_allowed(bot_allow: bool) -> bool:
    if tool_interruptions_disabled():
        return False
    return bot_allow
