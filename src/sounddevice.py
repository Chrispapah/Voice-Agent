"""Server-safe fallback for optional local audio features.

This project deploys in environments that do not provide PortAudio.
Vocode imports `sounddevice` eagerly through optional Vonage speaker code,
so this shim keeps server startup working while still failing loudly if any
local speaker or microphone feature is actually used.
"""

from __future__ import annotations


_ERROR_MESSAGE = (
    "sounddevice is unavailable in this deployment environment because "
    "PortAudio is not installed. Local speaker/microphone playback is not "
    "supported on the server."
)


class PortAudioError(RuntimeError):
    """Compatibility error type for callers expecting a sounddevice failure."""


class _UnavailableObject:
    def __init__(self, *args, **kwargs):
        raise PortAudioError(_ERROR_MESSAGE)

    def __getattr__(self, name: str):
        raise PortAudioError(_ERROR_MESSAGE)


class OutputStream(_UnavailableObject):
    pass


class InputStream(_UnavailableObject):
    pass


class Stream(_UnavailableObject):
    pass


default = _UnavailableObject


def query_devices(*args, **kwargs):
    raise PortAudioError(_ERROR_MESSAGE)


def play(*args, **kwargs):
    raise PortAudioError(_ERROR_MESSAGE)


def rec(*args, **kwargs):
    raise PortAudioError(_ERROR_MESSAGE)


def playrec(*args, **kwargs):
    raise PortAudioError(_ERROR_MESSAGE)


def wait(*args, **kwargs):
    raise PortAudioError(_ERROR_MESSAGE)


def stop(*args, **kwargs):
    raise PortAudioError(_ERROR_MESSAGE)


def __getattr__(name: str):
    raise PortAudioError(_ERROR_MESSAGE)
