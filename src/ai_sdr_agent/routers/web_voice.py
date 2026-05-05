from __future__ import annotations

import asyncio
import base64
import json
import uuid
from typing import Any
from urllib.parse import urlencode

import httpx
import websockets
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
        try:
            async with websockets.connect(
                uri,
                additional_headers=[("Authorization", f"Token {key}")],
                ping_interval=20,
                ping_timeout=20,
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
                            await dg.send(chunk)
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        logger.exception("Deepgram forward failed")

                forward_task = asyncio.create_task(forward_audio())
                try:
                    async for message in dg:
                        if stop_event.is_set():
                            break
                        if isinstance(message, bytes):
                            continue
                        try:
                            payload = json.loads(message)
                        except json.JSONDecodeError:
                            continue
                        if payload.get("type") == "Error":
                            err = payload.get("description") or str(payload)
                            await _send_json(websocket, {"type": "error", "message": f"Deepgram: {err}"})
                            continue
                        channel = payload.get("channel") or {}
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
                finally:
                    forward_task.cancel()
                    try:
                        await forward_task
                    except asyncio.CancelledError:
                        pass
                    try:
                        await dg.send(json.dumps({"type": "CloseStream"}))
                    except Exception:
                        pass
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Deepgram connection failed")
            await _send_json(websocket, {"type": "error", "message": "Could not connect to Deepgram"})

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
