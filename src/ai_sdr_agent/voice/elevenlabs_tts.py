from __future__ import annotations

import asyncio
import base64
from typing import Any, Awaitable, Callable
from urllib.parse import urlencode

import httpx
from loguru import logger

from ai_sdr_agent.text.greek_number_words import expand_for_greek_elevenlabs_tts

SendJson = Callable[[dict[str, Any]], Awaitable[None]]
ShouldContinue = Callable[[], bool]
MarkFirstAudio = Callable[[], None]


def elevenlabs_stream_url_and_json(
    voice_id: str,
    model_id: str,
    spoken_plain_text: str,
    *,
    optimize_streaming_latency: int,
    output_format: str = "mp3_22050_32",
) -> tuple[str, dict[str, Any]]:
    """ElevenLabs streaming TTS URL + JSON tuned for time-to-first-byte."""
    lat = max(0, min(4, int(optimize_streaming_latency)))
    q = urlencode(
        {
            "optimize_streaming_latency": str(lat),
            "output_format": output_format,
        }
    )
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream?{q}"
    tts_body_text = expand_for_greek_elevenlabs_tts(spoken_plain_text.strip())
    body: dict[str, Any] = {
        "text": tts_body_text,
        "model_id": model_id,
        # use_speaker_boost adds latency per ElevenLabs docs; style>0 adds compute.
        "voice_settings": {"use_speaker_boost": False, "style": 0.0},
    }
    return url, body


async def stream_elevenlabs_text_to_ws(
    spoken_text: str,
    *,
    httpx_client: httpx.AsyncClient,
    elevenlabs_api_key: str,
    voice_id: str,
    model_id: str,
    optimize_streaming_latency: int,
    send_json: SendJson,
    should_continue: ShouldContinue,
    mark_first_audio: MarkFirstAudio | None = None,
) -> bool:
    stripped = spoken_text.strip()
    if not stripped or not should_continue():
        return True
    if not elevenlabs_api_key or not voice_id:
        return True

    e_url, tts_json = elevenlabs_stream_url_and_json(
        voice_id,
        model_id,
        stripped,
        optimize_streaming_latency=optimize_streaming_latency,
    )
    headers = {
        "xi-api-key": elevenlabs_api_key,
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
    }
    try:
        async with httpx_client.stream(
            "POST",
            e_url,
            headers=headers,
            json=tts_json,
            timeout=120.0,
        ) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes(4096):
                if not should_continue():
                    return False
                if chunk:
                    if mark_first_audio is not None:
                        mark_first_audio()
                    await send_json(
                        {
                            "type": "agent.audio",
                            "chunk": base64.b64encode(chunk).decode("ascii"),
                        }
                    )
    except (httpx.HTTPError, asyncio.CancelledError) as exc:
        if isinstance(exc, asyncio.CancelledError):
            raise
        logger.exception("ElevenLabs streaming failed")
        await send_json({"type": "error", "message": "TTS request failed"})
        return False

    if should_continue():
        await send_json({"type": "agent.audio_segment_end"})
    return True
