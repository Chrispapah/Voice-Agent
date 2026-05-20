from __future__ import annotations

import asyncio
import base64
import json
from typing import Any, Awaitable, Callable
from urllib.parse import urlencode

import aiohttp
from loguru import logger

SendJson = Callable[[dict[str, Any]], Awaitable[None]]
OnTranscriptFinal = Callable[[str], Awaitable[None]]
OnSpeechStarted = Callable[[], Awaitable[None]]


def _realtime_url(model: str) -> str:
    return "wss://api.openai.com/v1/realtime?" + urlencode({"model": model})


class OpenAIRealtimeVoiceBridge:
    """Server-side bridge between browser PCM audio and OpenAI Realtime.

    The app's LangGraph service still owns the agent response. Realtime is used
    for server VAD/transcription and speech synthesis around that text turn.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        voice: str,
        transcription_model: str,
        instructions: str | None,
        send_json: SendJson,
        on_transcript_final: OnTranscriptFinal,
        on_speech_started: OnSpeechStarted,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.transcription_model = transcription_model
        self.instructions = instructions
        self.send_json = send_json
        self.on_transcript_final = on_transcript_final
        self.on_speech_started = on_speech_started
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._receive_task: asyncio.Task[None] | None = None
        self._response_done: asyncio.Future[None] | None = None

    async def connect(self) -> None:
        timeout = aiohttp.ClientTimeout(total=None, connect=30, sock_connect=30, sock_read=None)
        session = aiohttp.ClientSession(timeout=timeout)
        try:
            self._ws = await session.ws_connect(
                _realtime_url(self.model),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                },
                heartbeat=20,
                autoping=True,
            )
        except Exception:
            await session.close()
            raise
        self._receive_task = asyncio.create_task(self._receive_loop(session))
        await self._send_openai(
            {
                "type": "session.update",
                "session": {
                    "modalities": ["text", "audio"],
                    "instructions": self.instructions
                    or "You are a voice renderer. Transcribe user speech and speak provided assistant text naturally.",
                    "voice": self.voice,
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",
                    "turn_detection": {
                        "type": "server_vad",
                        "create_response": False,
                        "interrupt_response": True,
                    },
                    "input_audio_transcription": {"model": self.transcription_model},
                },
            }
        )

    async def close(self) -> None:
        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
        if self._ws and not self._ws.closed:
            await self._ws.close()

    async def append_audio(self, chunk: bytes) -> None:
        if not self._ws or self._ws.closed:
            return
        await self._send_openai(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(chunk).decode("ascii"),
            }
        )

    async def cancel_response(self) -> None:
        if not self._ws or self._ws.closed:
            return
        await self._send_openai({"type": "response.cancel"})
        if self._response_done and not self._response_done.done():
            self._response_done.set_result(None)

    async def speak_text(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped or not self._ws or self._ws.closed:
            return True
        loop = asyncio.get_running_loop()
        done = loop.create_future()
        self._response_done = done
        await self._send_openai(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": f"Read this exact assistant message aloud without adding or changing words:\n\n{stripped}",
                        }
                    ],
                },
            }
        )
        await self._send_openai(
            {
                "type": "response.create",
                "response": {
                    "modalities": ["audio"],
                    "voice": self.voice,
                    "instructions": "Speak only the exact text from the latest message. Do not add commentary.",
                },
            }
        )
        try:
            await asyncio.wait_for(done, timeout=120.0)
        except asyncio.TimeoutError:
            logger.warning("OpenAI Realtime TTS timed out")
            await self.send_json({"type": "error", "message": "OpenAI Realtime speech timed out"})
            return False
        return True

    async def _send_openai(self, payload: dict[str, Any]) -> None:
        if not self._ws or self._ws.closed:
            return
        await self._ws.send_str(json.dumps(payload))

    async def _receive_loop(self, session: aiohttp.ClientSession) -> None:
        try:
            assert self._ws is not None
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    await self._handle_event(json.loads(msg.data))
                elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSING, aiohttp.WSMsgType.CLOSED):
                    break
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.warning("OpenAI Realtime websocket error: {}", msg)
                    break
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("OpenAI Realtime receive loop failed")
            await self.send_json({"type": "error", "message": "OpenAI Realtime connection failed"})
        finally:
            if self._response_done and not self._response_done.done():
                self._response_done.set_result(None)
            await session.close()

    async def _handle_event(self, event: dict[str, Any]) -> None:
        event_type = str(event.get("type") or "")
        if event_type == "error":
            error = event.get("error") if isinstance(event.get("error"), dict) else {}
            message = error.get("message") or "OpenAI Realtime returned an error"
            await self.send_json({"type": "error", "message": str(message)})
            if self._response_done and not self._response_done.done():
                self._response_done.set_result(None)
            return
        if event_type == "input_audio_buffer.speech_started":
            await self.on_speech_started()
            return
        if event_type == "conversation.item.input_audio_transcription.delta":
            delta = str(event.get("delta") or "")
            if delta:
                await self.send_json({"type": "transcript.partial", "text": delta})
            return
        if event_type in {
            "conversation.item.input_audio_transcription.completed",
            "conversation.item.input_audio_transcription.done",
        }:
            transcript = str(event.get("transcript") or "").strip()
            if transcript:
                await self.send_json({"type": "transcript.final", "text": transcript})
                await self.on_transcript_final(transcript)
            return
        if event_type == "response.audio.delta":
            delta = str(event.get("delta") or "")
            if delta:
                await self.send_json({"type": "agent.audio", "chunk": delta, "format": "pcm16", "sample_rate": 24000})
            return
        if event_type == "response.audio.done":
            await self.send_json({"type": "agent.audio_segment_end", "format": "pcm16", "sample_rate": 24000})
            return
        if event_type in {"response.done", "response.cancelled"}:
            if self._response_done and not self._response_done.done():
                self._response_done.set_result(None)
