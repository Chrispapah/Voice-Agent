from __future__ import annotations

import asyncio
import base64
import json
import uuid
from typing import Any
from urllib.parse import urlencode

import aiohttp
import httpx
from aiohttp import WSMsgType
from aiohttp.client_exceptions import WSServerHandshakeError
from fastapi import APIRouter, HTTPException, WebSocket
from jose import JWTError
from loguru import logger
from starlette.websockets import WebSocketDisconnect

from ai_sdr_agent.auth.dependencies import decode_supabase_jwt
from ai_sdr_agent.config import SDRSettings, get_settings
from ai_sdr_agent.db.engine import get_async_session_factory
from ai_sdr_agent.db.repositories import (
    PgCallLogRepository,
    PgLeadRepository,
    PgSessionStore,
)
from ai_sdr_agent.routers.test_sessions import _build_service_for_bot, _verify_bot

router = APIRouter(prefix="/api/bots", tags=["voice"])


def _merge_voice_credentials(bot_cfg: dict[str, Any], settings: SDRSettings) -> dict[str, Any]:
    out = dict(bot_cfg)
    if not out.get("deepgram_api_key") and settings.deepgram_api_key:
        out["deepgram_api_key"] = settings.deepgram_api_key
    if not out.get("elevenlabs_api_key") and settings.elevenlabs_api_key:
        out["elevenlabs_api_key"] = settings.elevenlabs_api_key
    if not out.get("elevenlabs_voice_id") and settings.elevenlabs_voice_id:
        out["elevenlabs_voice_id"] = settings.elevenlabs_voice_id
    return out


def _deepgram_listen_url(*, model: str, language: str) -> str:
    params = {
        "model": model,
        "language": language,
        "smart_format": "true",
        "interim_results": "true",
        "endpointing": "400",
        "vad_events": "true",
    }
    return "wss://api.deepgram.com/v1/listen?" + urlencode(params)


def _first_deepgram_transcript_channel(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Resolve the channel object that holds ``alternatives``.

    Deepgram v1 sends several JSON message types on the same socket. Only
    ``Results`` has ``channel`` as an object with ``alternatives``. Other types
    (e.g. ``SpeechStarted``, ``UtteranceEnd``) use ``channel`` as a list of ints,
    which must not be passed to ``.get``.
    """
    ch = payload.get("channel")
    if isinstance(ch, dict):
        return ch
    if isinstance(ch, list) and ch and isinstance(ch[0], dict):
        return ch[0]
    multi = payload.get("channels")
    if isinstance(multi, list) and multi and isinstance(multi[0], dict):
        return multi[0]
    return None


def _client_message_for_deepgram_connect_failure(exc: Exception) -> str:
    """Map connect-time failures to hints; full detail stays in server logs."""
    msg_l = str(exc).lower()
    if isinstance(exc, TimeoutError):
        return (
            "Connection to Deepgram timed out. Allow outbound HTTPS/WSS (TCP 443) from this host, or try another network."
        )
    if isinstance(exc, OSError):
        winerr = getattr(exc, "winerror", None)
        if winerr in (11001, 11002):
            return (
                "DNS could not resolve api.deepgram.com (Windows). Check DNS, VPN, or try "
                "nslookup api.deepgram.com from the machine running the API."
            )
    if "getaddrinfo" in msg_l or "name or service not known" in msg_l or "nodename nor servname" in msg_l:
        return (
            "DNS could not resolve api.deepgram.com. Verify internet/VPN and DNS; from the server run: "
            "nslookup api.deepgram.com (or ping api.deepgram.com)."
        )
    if any(x in msg_l for x in ("certificate", "ssl", "tls", "cert verify")):
        return (
            "TLS error connecting to Deepgram. Check system clock, antivirus HTTPS scanning, and corporate SSL inspection."
        )
    if "connection refused" in msg_l or "network is unreachable" in msg_l:
        return (
            "Could not reach Deepgram on the network. Confirm firewall allows outbound WSS to api.deepgram.com:443."
        )
    if isinstance(exc, TypeError) and "additional_headers" in msg_l and "create_connection" in msg_l:
        return (
            "WebSocket client bug: upgrade/redeploy the API (Deepgram uses legacy websockets.connect with extra_headers)."
        )
    return (
        "Could not open Deepgram. See API logs for the exception; confirm outbound access to wss://api.deepgram.com."
    )


async def _send_json(ws: WebSocket, payload: dict[str, Any]) -> None:
    try:
        await ws.send_text(json.dumps(payload))
    except Exception as exc:  # pragma: no cover
        logger.debug("web_voice send failed: {}", exc)


@router.websocket("/{bot_id}/voice-session")
async def voice_session(websocket: WebSocket, bot_id: str) -> None:
    await websocket.accept()
    settings = get_settings()
    user_id: uuid.UUID | None = None
    conversation_id: str | None = None
    bot_cfg_merged: dict[str, Any] | None = None
    bid: uuid.UUID | None = None
    audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
    stop_event = asyncio.Event()
    generation = 0
    active_pipeline: asyncio.Task[None] | None = None
    dg_tasks: list[asyncio.Task[None]] = []
    httpx_client = httpx.AsyncClient()

    def invalidate_turns() -> int:
        nonlocal generation
        generation += 1
        return generation

    async def close_deepgram_tasks() -> None:
        stop_event.set()
        for t in dg_tasks:
            if not t.done():
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass
        dg_tasks.clear()
        stop_event.clear()

    async def pipeline(user_text: str, my_gen: int) -> None:
        nonlocal conversation_id, bot_cfg_merged, bid
        assert conversation_id and bot_cfg_merged and bid
        if not user_text.strip():
            return
        try:
            async with get_async_session_factory()() as db:
                lead_repo = PgLeadRepository(db)
                session_store = PgSessionStore(db, bid)
                call_log_repo = PgCallLogRepository(db, bid)
                svc = _build_service_for_bot(bot_cfg_merged, lead_repo, session_store, call_log_repo)
                state = await svc.handle_turn(conversation_id, user_text)
                await db.commit()
            if my_gen != generation:
                return
            reply = (state.get("last_agent_response") or "").strip()
            await _send_json(
                websocket,
                {
                    "type": "agent.text",
                    "text": reply,
                    "stage": state.get("current_node"),
                    "next_node": state.get("next_node"),
                },
            )
            if not reply or my_gen != generation:
                await _send_json(websocket, {"type": "agent.done"})
                return
            eleven_key = bot_cfg_merged.get("elevenlabs_api_key") or ""
            voice_id = bot_cfg_merged.get("elevenlabs_voice_id") or ""
            model_id = str(bot_cfg_merged.get("elevenlabs_model_id") or "eleven_turbo_v2")
            if not eleven_key or not voice_id:
                await _send_json(
                    websocket,
                    {"type": "error", "message": "ElevenLabs API key or voice id not configured."},
                )
                await _send_json(websocket, {"type": "agent.done"})
                return
            try:
                e_url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
                e_headers = {
                    "xi-api-key": eleven_key,
                    "Accept": "audio/mpeg",
                    "Content-Type": "application/json",
                }
                async with httpx_client.stream(
                    "POST",
                    e_url,
                    headers=e_headers,
                    json={"text": reply, "model_id": model_id},
                    timeout=120.0,
                ) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_bytes(4096):
                        if my_gen != generation:
                            break
                        if chunk:
                            await _send_json(
                                websocket,
                                {
                                    "type": "agent.audio",
                                    "chunk": base64.b64encode(chunk).decode("ascii"),
                                },
                            )
            except (httpx.HTTPError, asyncio.CancelledError) as exc:
                if isinstance(exc, asyncio.CancelledError):
                    raise
                logger.exception("ElevenLabs streaming failed")
                await _send_json(websocket, {"type": "error", "message": "TTS request failed"})
            if my_gen == generation:
                await _send_json(websocket, {"type": "agent.done"})
        except asyncio.CancelledError:
            await _send_json(websocket, {"type": "agent.interrupted"})
            raise

    async def schedule_pipeline(user_text: str) -> None:
        nonlocal active_pipeline, generation
        if active_pipeline and not active_pipeline.done():
            active_pipeline.cancel()
            try:
                await active_pipeline
            except asyncio.CancelledError:
                pass
        my_gen = invalidate_turns()
        active_pipeline = asyncio.create_task(pipeline(user_text, my_gen))

    async def run_deepgram() -> None:
        assert bot_cfg_merged is not None
        key = (bot_cfg_merged.get("deepgram_api_key") or "").strip()
        if not key:
            await _send_json(
                websocket,
                {"type": "error", "message": "Deepgram API key required for browser voice (bot or server env)." },
            )
            return
        model = str(bot_cfg_merged.get("deepgram_model") or "nova-2")
        language = str(bot_cfg_merged.get("deepgram_language") or "en-US")
        uri = _deepgram_listen_url(model=model, language=language)
        dg_timeout = aiohttp.ClientTimeout(total=None, connect=30, sock_connect=30, sock_read=None)
        try:
            # aiohttp WebSocket: avoids fragile interactions between the `websockets` package,
            # vocode's websockets<13 pin, and uvicorn/uvloop on Python 3.12.
            async with aiohttp.ClientSession(timeout=dg_timeout) as session:
                async with session.ws_connect(
                    uri,
                    headers={"Authorization": f"Token {key}"},
                    heartbeat=20,
                    autoping=True,
                ) as dg:
                    async def forward_audio() -> None:
                        try:
                            while not stop_event.is_set():
                                try:
                                    chunk = await asyncio.wait_for(audio_queue.get(), timeout=0.25)
                                except asyncio.TimeoutError:
                                    continue
                                if chunk is None:
                                    break
                                await dg.send_bytes(chunk)
                        except asyncio.CancelledError:
                            raise
                        except Exception:
                            logger.exception("Deepgram forward failed")

                    forward_task = asyncio.create_task(forward_audio())
                    try:
                        async for msg in dg:
                            if stop_event.is_set():
                                break
                            if msg.type == WSMsgType.BINARY:
                                continue
                            if msg.type == WSMsgType.TEXT:
                                try:
                                    payload = json.loads(msg.data)
                                except json.JSONDecodeError:
                                    continue
                                if payload.get("type") == "Error":
                                    err = payload.get("description") or str(payload)
                                    await _send_json(websocket, {"type": "error", "message": f"Deepgram: {err}"})
                                    continue
                                channel = _first_deepgram_transcript_channel(payload)
                                if channel is None:
                                    continue
                                alts = channel.get("alternatives") or []
                                if not alts:
                                    continue
                                transcript = (alts[0].get("transcript") or "").strip()
                                if not transcript:
                                    continue
                                is_final = bool(payload.get("is_final") or payload.get("speech_final"))
                                if is_final:
                                    await _send_json(websocket, {"type": "transcript.final", "text": transcript})
                                    asyncio.create_task(schedule_pipeline(transcript))
                                else:
                                    await _send_json(websocket, {"type": "transcript.partial", "text": transcript})
                            elif msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSING):
                                break
                            elif msg.type == WSMsgType.ERROR:
                                logger.warning("Deepgram websocket protocol error: {}", msg)
                                break
                    finally:
                        forward_task.cancel()
                        try:
                            await forward_task
                        except asyncio.CancelledError:
                            pass
                        try:
                            await dg.send_str(json.dumps({"type": "CloseStream"}))
                        except Exception:
                            pass
        except asyncio.CancelledError:
            raise
        except WSServerHandshakeError as exc:
            code = exc.status
            logger.warning("Deepgram WebSocket handshake failed HTTP {}", code)
            if code == 401:
                msg = (
                    "Deepgram rejected the API key (401). Add DEEPGRAM_API_KEY on the server (Railway) "
                    "or a real deepgram_api_key on the bot in Supabase (not a masked placeholder)."
                )
            elif code in (402, 403):
                msg = f"Deepgram returned HTTP {code}; check billing, project, or key permissions."
            else:
                msg = f"Deepgram WebSocket rejected the connection (HTTP {code}). See server logs."
            await _send_json(websocket, {"type": "error", "message": msg})
        except aiohttp.ClientError as exc:
            logger.exception("Deepgram connection failed (aiohttp)")
            await _send_json(
                websocket,
                {"type": "error", "message": _client_message_for_deepgram_connect_failure(exc)},
            )
        except Exception as exc:
            logger.exception("Deepgram connection failed")
            await _send_json(
                websocket,
                {"type": "error", "message": _client_message_for_deepgram_connect_failure(exc)},
            )

    try:
        raw = await websocket.receive_text()
        try:
            hello = json.loads(raw)
        except json.JSONDecodeError:
            await _send_json(websocket, {"type": "error", "message": "Expected JSON auth message"})
            await websocket.close(code=4400)
            return
        if hello.get("type") != "auth" or not hello.get("access_token"):
            await _send_json(websocket, {"type": "error", "message": "First message must be auth with access_token"})
            await websocket.close(code=4401)
            return
        try:
            payload = decode_supabase_jwt(str(hello["access_token"]))
        except JWTError:
            await _send_json(websocket, {"type": "error", "message": "Invalid or expired token"})
            await websocket.close(code=4401)
            return
        sub = payload.get("sub")
        if not sub:
            await _send_json(websocket, {"type": "error", "message": "Invalid token payload"})
            await websocket.close(code=4401)
            return
        user_id = uuid.UUID(str(sub))

        raw2 = await websocket.receive_text()
        start_msg = json.loads(raw2)
        if start_msg.get("type") != "session.start":
            await _send_json(websocket, {"type": "error", "message": "Expected session.start"})
            await websocket.close(code=4400)
            return
        lead_id = str(start_msg.get("lead_id") or "").strip()
        if not lead_id:
            await _send_json(websocket, {"type": "error", "message": "lead_id is required"})
            await websocket.close(code=4400)
            return
        existing_id = start_msg.get("conversation_id")
        existing_id = str(existing_id).strip() if existing_id else None

        async with get_async_session_factory()() as db:
            try:
                bot = await _verify_bot(bot_id, user_id, db)
            except HTTPException as exc:
                await _send_json(websocket, {"type": "error", "message": str(exc.detail)})
                await websocket.close(code=1008)
                return
            bid = bot.id
            bot_cfg_merged = _merge_voice_credentials(bot.to_config_dict(), settings)
            lead_repo = PgLeadRepository(db)
            session_store = PgSessionStore(db, bid)
            call_log_repo = PgCallLogRepository(db, bid)
            svc = _build_service_for_bot(bot_cfg_merged, lead_repo, session_store, call_log_repo)

            if existing_id:
                row = await session_store.get(existing_id)
                if row is None:
                    await _send_json(websocket, {"type": "error", "message": "Unknown conversation_id"})
                    await websocket.close(code=4404)
                    return
                conversation_id = existing_id
            else:
                conversation_id = await svc.start_session(lead_id, bot_config=bot_cfg_merged)
                state0 = await svc.handle_turn(conversation_id, "")
                await db.commit()
                my_gen = invalidate_turns()
                active_pipeline = asyncio.create_task(
                    _send_initial_greeting_tts(
                        websocket,
                        state0,
                        bot_cfg_merged,
                        httpx_client,
                        my_gen,
                        lambda: generation,
                    )
                )

        await _send_json(websocket, {"type": "ready", "conversation_id": conversation_id})

        dg_task = asyncio.create_task(run_deepgram())
        dg_tasks.append(dg_task)

        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            if message.get("type") != "websocket.receive":
                break
            if "bytes" in message and message["bytes"] is not None:
                await audio_queue.put(message["bytes"])
                continue
            if "text" in message and message["text"] is not None:
                try:
                    ctrl = json.loads(message["text"])
                except json.JSONDecodeError:
                    continue
                ctype = ctrl.get("type")
                if ctype == "interrupt":
                    invalidate_turns()
                    if active_pipeline and not active_pipeline.done():
                        active_pipeline.cancel()
                        try:
                            await active_pipeline
                        except asyncio.CancelledError:
                            pass
                    await _send_json(websocket, {"type": "agent.interrupted"})
                elif ctype == "ping":
                    await _send_json(websocket, {"type": "pong"})
    except WebSocketDisconnect:
        pass
    except asyncio.CancelledError:
        pass
    except Exception:
        logger.exception("voice_session handler error")
        try:
            await _send_json(websocket, {"type": "error", "message": "Internal server error"})
        except Exception:
            pass
    finally:
        await audio_queue.put(None)
        await close_deepgram_tasks()
        if active_pipeline and not active_pipeline.done():
            active_pipeline.cancel()
            try:
                await active_pipeline
            except asyncio.CancelledError:
                pass
        await httpx_client.aclose()


async def _send_initial_greeting_tts(
    websocket: WebSocket,
    state0: dict[str, Any],
    bot_cfg_merged: dict[str, Any],
    httpx_client: httpx.AsyncClient,
    my_gen: int,
    current_generation: Any,
) -> None:
    """Play initial greeting for a brand-new session (already persisted)."""
    try:
        reply = (state0.get("last_agent_response") or "").strip()
        await _send_json(
            websocket,
            {
                "type": "agent.text",
                "text": reply,
                "stage": state0.get("current_node"),
                "next_node": state0.get("next_node"),
            },
        )
        if not reply:
            await _send_json(websocket, {"type": "agent.done"})
            return
        eleven_key = bot_cfg_merged.get("elevenlabs_api_key") or ""
        voice_id = bot_cfg_merged.get("elevenlabs_voice_id") or ""
        model_id = str(bot_cfg_merged.get("elevenlabs_model_id") or "eleven_turbo_v2")
        if not eleven_key or not voice_id:
            await _send_json(websocket, {"type": "agent.done"})
            return
        e_url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
        e_headers = {
            "xi-api-key": eleven_key,
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
        }
        async with httpx_client.stream(
            "POST",
            e_url,
            headers=e_headers,
            json={"text": reply, "model_id": model_id},
            timeout=120.0,
        ) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes(4096):
                if my_gen != current_generation():
                    break
                if chunk:
                    await _send_json(
                        websocket,
                        {
                            "type": "agent.audio",
                            "chunk": base64.b64encode(chunk).decode("ascii"),
                        },
                    )
        if my_gen == current_generation():
            await _send_json(websocket, {"type": "agent.done"})
    except httpx.HTTPError:
        logger.exception("Initial greeting TTS failed")
        await _send_json(websocket, {"type": "agent.done"})
    except asyncio.CancelledError:
        await _send_json(websocket, {"type": "agent.interrupted"})
        raise
