from __future__ import annotations

import asyncio
import hashlib
import json
import secrets
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import aiohttp
import httpx
from aiohttp import WSMsgType
from aiohttp.client_exceptions import WSServerHandshakeError
from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, status
from loguru import logger
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.websockets import WebSocketDisconnect

from ai_sdr_agent.auth.dependencies import get_current_user_id
from ai_sdr_agent.config import get_settings
from ai_sdr_agent.db.engine import get_async_session, get_async_session_factory
from ai_sdr_agent.db.models import AgentPreviewShareRow, BotConfigRow, LeadRow
from ai_sdr_agent.db.repositories import PgCallLogRepository, PgLeadRepository, PgSessionStore
from ai_sdr_agent.routers.test_sessions import _build_service_for_bot, _verify_bot
from ai_sdr_agent.routers.web_voice import (
    _client_message_for_deepgram_connect_failure,
    _deepgram_listen_url,
    _first_deepgram_transcript_channel,
    _merge_voice_credentials,
    _send_initial_greeting_tts,
    _send_json,
)
from ai_sdr_agent.transcriber_factory import (
    normalize_deepgram_language_code,
    prefer_nova3_for_greek_browser_stt,
    resolve_web_voice_deepgram_model,
)
from ai_sdr_agent.voice.elevenlabs_tts import stream_elevenlabs_text_to_ws
from ai_sdr_agent.voice.echo_filter import RealtimeEchoGuard
from ai_sdr_agent.voice.openai_realtime import OpenAIRealtimeVoiceBridge
from ai_sdr_agent.voice.turn_orchestrator import run_voice_graph_turn

router = APIRouter(tags=["agent-previews"])


class AgentPreviewShareCreateRequest(BaseModel):
    expires_in_days: int | None = Field(default=30, ge=1, le=365)
    max_sessions: int = Field(default=100, ge=1, le=10_000)
    title: str | None = Field(default=None, max_length=200)
    welcome_message: str | None = Field(default=None, max_length=500)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _preview_path(token: str) -> str:
    return f"/preview/agent/{token}"


def _share_response(row: AgentPreviewShareRow, token: str, request: Request) -> dict[str, Any]:
    origin = request.headers.get("origin") or str(request.base_url).rstrip("/")
    path = _preview_path(token)
    return {
        "id": str(row.id),
        "bot_id": str(row.bot_id),
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "expires_at": row.expires_at.isoformat() if row.expires_at else None,
        "revoked_at": row.revoked_at.isoformat() if row.revoked_at else None,
        "max_sessions": row.max_sessions,
        "session_count": row.session_count,
        "token": token,
        "preview_path": path,
        "preview_url": f"{origin}{path}",
    }


def _public_share_payload(row: AgentPreviewShareRow, bot: BotConfigRow) -> dict[str, Any]:
    return {
        "id": str(row.id),
        "agent_name": bot.name,
        "title": row.title or f"Talk to {bot.name}",
        "welcome_message": row.welcome_message or "Start a live voice conversation with this AI agent.",
        "voice_provider": bot.voice_provider,
        "expires_at": row.expires_at.isoformat() if row.expires_at else None,
        "remaining_sessions": max(0, row.max_sessions - row.session_count),
    }


async def _get_active_preview_share(
    token: str,
    session: AsyncSession,
    *,
    require_capacity: bool = True,
) -> tuple[AgentPreviewShareRow, BotConfigRow]:
    result = await session.execute(
        select(AgentPreviewShareRow, BotConfigRow)
        .join(BotConfigRow, AgentPreviewShareRow.bot_id == BotConfigRow.id)
        .where(AgentPreviewShareRow.token_hash == _token_hash(token))
    )
    row = result.one_or_none()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Preview link not found")
    share, bot = row
    now = _utcnow()
    if share.revoked_at is not None or (share.expires_at is not None and share.expires_at <= now):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Preview link not found")
    if require_capacity and share.session_count >= share.max_sessions:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Preview session limit reached")
    return share, bot


@router.post("/api/bots/{bot_id}/agent-preview-share", status_code=status.HTTP_201_CREATED)
async def create_agent_preview_share(
    bot_id: str,
    body: AgentPreviewShareCreateRequest,
    request: Request,
    user_id: uuid.UUID = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_async_session),
):
    bot = await _verify_bot(bot_id, user_id, session)
    token = secrets.token_urlsafe(32)
    row = AgentPreviewShareRow(
        bot_id=bot.id,
        token_hash=_token_hash(token),
        created_by=user_id,
        expires_at=_utcnow() + timedelta(days=body.expires_in_days or 30),
        max_sessions=body.max_sessions,
        title=body.title,
        welcome_message=body.welcome_message,
    )
    session.add(row)
    await session.commit()
    return _share_response(row, token, request)


@router.post("/api/agent-preview-shares/{share_id}/revoke")
async def revoke_agent_preview_share(
    share_id: str,
    user_id: uuid.UUID = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_async_session),
):
    try:
        parsed_share_id = uuid.UUID(share_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Preview share not found") from exc
    result = await session.execute(
        select(AgentPreviewShareRow)
        .join(BotConfigRow, AgentPreviewShareRow.bot_id == BotConfigRow.id)
        .where(AgentPreviewShareRow.id == parsed_share_id, BotConfigRow.user_id == user_id)
    )
    row = result.scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Preview share not found")
    row.revoked_at = _utcnow()
    await session.commit()
    return {"id": str(row.id), "revoked_at": row.revoked_at.isoformat()}


@router.get("/api/public/agent-previews/{token}")
async def get_public_agent_preview(
    token: str,
    session: AsyncSession = Depends(get_async_session),
):
    share, bot = await _get_active_preview_share(token, session, require_capacity=False)
    return _public_share_payload(share, bot)


@router.post("/api/public/agent-previews/{token}/session", status_code=status.HTTP_201_CREATED)
async def start_public_agent_preview_session(
    token: str,
    session: AsyncSession = Depends(get_async_session),
):
    share, bot = await _get_active_preview_share(token, session)
    from ai_sdr_agent.db.repositories import PgLeadRepository

    lead_repo = PgLeadRepository(session)
    preview_phone = f"preview-share-{share.id}"
    lead = await lead_repo.get_by_bot_and_phone(bot.id, preview_phone)
    if lead is None:
        lead = LeadRow(
            bot_id=bot.id,
            lead_name="Visitor",
            company="",
            phone_number=preview_phone,
            lead_email="",
            lead_context="",
            lifecycle_stage="preview",
            timezone="UTC",
            owner_name="",
            calendar_id="",
            metadata_json={"source": "agent_preview", "share_id": str(share.id)},
        )
        session.add(lead)
        await session.flush()
    share.session_count += 1
    await session.commit()
    return {
        "lead_id": str(lead.id),
        "conversation_id": None,
        "voice_provider": bot.voice_provider,
    }


@router.websocket("/api/public/agent-previews/{token}/voice-session")
async def public_agent_preview_voice_session(websocket: WebSocket, token: str) -> None:
    """Public Deepgram + ElevenLabs voice preview.

    This mirrors the authenticated browser voice protocol, but validates a
    scoped preview token from the URL instead of a Supabase JWT from the client.
    """
    await websocket.accept()
    settings = get_settings()
    conversation_id: str | None = None
    bot_cfg_merged: dict[str, Any] | None = None
    bid: uuid.UUID | None = None
    audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
    stop_event = asyncio.Event()
    generation = 0
    active_pipeline: asyncio.Task[None] | None = None
    dg_tasks: list[asyncio.Task[None]] = []
    httpx_client = httpx.AsyncClient()

    from ai_sdr_agent.services.tool_context import voice_interruptions_allowed

    def allow_voice_interruptions() -> bool:
        return voice_interruptions_allowed(
            bool((bot_cfg_merged or {}).get("allow_voice_interruptions", True))
        )

    def invalidate_turns() -> int:
        nonlocal generation
        generation += 1
        return generation

    async def close_deepgram_tasks() -> None:
        stop_event.set()
        for task in dg_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        dg_tasks.clear()
        stop_event.clear()

    async def pipeline(user_text: str, my_gen: int, *, stt_final_pc: float) -> None:
        nonlocal conversation_id, bot_cfg_merged, bid
        assert conversation_id and bot_cfg_merged and bid
        eleven_key = bot_cfg_merged.get("elevenlabs_api_key") or ""
        voice_id = bot_cfg_merged.get("elevenlabs_voice_id") or ""
        model_id = str(bot_cfg_merged.get("elevenlabs_model_id") or "eleven_turbo_v2")
        tts_latency_opt = int(get_settings().elevenlabs_optimize_streaming_latency)

        async def _stream_elevenlabs_text(spoken_text: str, mark_first_audio: Any) -> bool:
            return await stream_elevenlabs_text_to_ws(
                spoken_text,
                httpx_client=httpx_client,
                elevenlabs_api_key=eleven_key,
                voice_id=voice_id,
                model_id=model_id,
                optimize_streaming_latency=tts_latency_opt,
                send_json=lambda payload: _send_json(websocket, payload),
                should_continue=lambda: my_gen == generation,
                mark_first_audio=mark_first_audio,
            )

        await run_voice_graph_turn(
            bot_id=str(bid),
            bid=bid,
            bot_cfg=bot_cfg_merged,
            conversation_id=conversation_id,
            user_text=user_text,
            stt_final_pc=stt_final_pc,
            send_json=lambda payload: _send_json(websocket, payload),
            synthesize_text=_stream_elevenlabs_text,
            has_speech_output=lambda: bool(eleven_key and voice_id),
            is_current=lambda: my_gen == generation,
        )

    async def schedule_pipeline(user_text: str, *, stt_final_pc: float) -> None:
        nonlocal active_pipeline
        if active_pipeline and not active_pipeline.done():
            active_pipeline.cancel()
            try:
                await active_pipeline
            except asyncio.CancelledError:
                pass
        my_gen = invalidate_turns()
        active_pipeline = asyncio.create_task(pipeline(user_text, my_gen, stt_final_pc=stt_final_pc))

    async def run_deepgram() -> None:
        assert bot_cfg_merged is not None
        key = (bot_cfg_merged.get("deepgram_api_key") or "").strip()
        if not key:
            await _send_json(websocket, {"type": "error", "message": "Deepgram API key required for live preview."})
            return
        language = normalize_deepgram_language_code(str(bot_cfg_merged.get("deepgram_language") or "el"))
        model = resolve_web_voice_deepgram_model(str(bot_cfg_merged.get("deepgram_model") or "nova-2"))
        model = prefer_nova3_for_greek_browser_stt(model, language)
        uri = _deepgram_listen_url(
            model=model,
            language=language,
            endpointing_ms=get_settings().deepgram_vad_threshold_ms,
        )
        dg_timeout = aiohttp.ClientTimeout(total=None, connect=30, sock_connect=30, sock_read=None)
        try:
            async with aiohttp.ClientSession(timeout=dg_timeout) as aio_session:
                async with aio_session.ws_connect(
                    uri,
                    headers={"Authorization": f"Token {key}"},
                    heartbeat=20,
                    autoping=True,
                ) as dg:
                    async def forward_audio() -> None:
                        while not stop_event.is_set():
                            try:
                                chunk = await asyncio.wait_for(audio_queue.get(), timeout=0.25)
                            except asyncio.TimeoutError:
                                continue
                            if chunk is None:
                                break
                            await dg.send_bytes(chunk)

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
                                    await _send_json(
                                        websocket,
                                        {"type": "error", "message": f"Deepgram: {payload.get('description') or payload}"},
                                    )
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
                                    stt_pc = time.perf_counter()
                                    logger.info(
                                        "Deepgram transcript_final route=agent_preview bot_id={} conversation_id={} text={!r}",
                                        bid,
                                        conversation_id,
                                        transcript,
                                    )
                                    await _send_json(websocket, {"type": "transcript.final", "text": transcript})
                                    if not allow_voice_interruptions() and active_pipeline and not active_pipeline.done():
                                        continue
                                    asyncio.create_task(schedule_pipeline(transcript, stt_final_pc=stt_pc))
                                else:
                                    await _send_json(websocket, {"type": "transcript.partial", "text": transcript})
                            elif msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSING):
                                break
                            elif msg.type == WSMsgType.ERROR:
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
            await _send_json(websocket, {"type": "error", "message": f"Deepgram WebSocket rejected the connection (HTTP {exc.status})."})
        except aiohttp.ClientError as exc:
            await _send_json(websocket, {"type": "error", "message": _client_message_for_deepgram_connect_failure(exc)})
        except Exception as exc:
            await _send_json(websocket, {"type": "error", "message": _client_message_for_deepgram_connect_failure(exc)})

    try:
        raw = await websocket.receive_text()
        start_msg = json.loads(raw)
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
                _, bot = await _get_active_preview_share(token, db, require_capacity=False)
            except HTTPException as exc:
                await _send_json(websocket, {"type": "error", "message": str(exc.detail)})
                await websocket.close(code=1008)
                return
            lead = await db.get(LeadRow, uuid.UUID(lead_id))
            if lead is None or lead.bot_id != bot.id:
                await _send_json(websocket, {"type": "error", "message": "Preview session not found"})
                await websocket.close(code=4404)
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

        await _send_json(
            websocket,
            {
                "type": "ready",
                "conversation_id": conversation_id,
                "allow_interruptions": allow_voice_interruptions(),
            },
        )
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
                if ctrl.get("type") == "interrupt":
                    if not allow_voice_interruptions():
                        continue
                    invalidate_turns()
                    if active_pipeline and not active_pipeline.done():
                        active_pipeline.cancel()
                        try:
                            await active_pipeline
                        except asyncio.CancelledError:
                            pass
                    await _send_json(websocket, {"type": "agent.interrupted"})
                elif ctrl.get("type") == "ping":
                    await _send_json(websocket, {"type": "pong"})
    except WebSocketDisconnect:
        pass
    except asyncio.CancelledError:
        pass
    except Exception:
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


@router.websocket("/api/public/agent-previews/{token}/voice-session/openai-realtime")
async def public_agent_preview_openai_realtime_voice_session(websocket: WebSocket, token: str) -> None:
    await websocket.accept()
    settings = get_settings()
    conversation_id: str | None = None
    bot_cfg_merged: dict[str, Any] | None = None
    bid: uuid.UUID | None = None
    generation = 0
    active_pipeline: asyncio.Task[None] | None = None
    bridge: OpenAIRealtimeVoiceBridge | None = None
    echo_guard = RealtimeEchoGuard()

    from ai_sdr_agent.services.tool_context import voice_interruptions_allowed

    def allow_voice_interruptions() -> bool:
        return voice_interruptions_allowed(
            bool((bot_cfg_merged or {}).get("allow_voice_interruptions", True))
        )

    def invalidate_turns() -> int:
        nonlocal generation
        generation += 1
        return generation

    async def cancel_active_turn(*, notify_client: bool = True) -> None:
        invalidate_turns()
        if active_pipeline and not active_pipeline.done():
            active_pipeline.cancel()
            try:
                await active_pipeline
            except asyncio.CancelledError:
                pass
        if bridge is not None:
            await bridge.cancel_response()
        if notify_client:
            await _send_json(websocket, {"type": "agent.interrupted"})

    async def synthesize_realtime_text(spoken_text: str, mark_first_audio: Any) -> bool:
        if bridge is None:
            return False
        echo_guard.record_agent_speech(spoken_text)
        ok = await bridge.speak_text(spoken_text)
        if ok:
            mark_first_audio()
        return ok

    async def pipeline(user_text: str, my_gen: int, *, stt_final_pc: float) -> None:
        nonlocal conversation_id, bot_cfg_merged, bid
        assert conversation_id and bot_cfg_merged and bid
        await run_voice_graph_turn(
            bot_id=str(bid),
            bid=bid,
            bot_cfg=bot_cfg_merged,
            conversation_id=conversation_id,
            user_text=user_text,
            stt_final_pc=stt_final_pc,
            send_json=lambda payload: _send_json(websocket, payload),
            synthesize_text=synthesize_realtime_text,
            has_speech_output=lambda: bridge is not None,
            is_current=lambda: my_gen == generation,
        )

    async def schedule_pipeline(user_text: str, *, stt_final_pc: float) -> None:
        nonlocal active_pipeline
        await cancel_active_turn(notify_client=False)
        my_gen = invalidate_turns()
        active_pipeline = asyncio.create_task(pipeline(user_text, my_gen, stt_final_pc=stt_final_pc))

    async def on_realtime_transcript_final(text: str) -> None:
        echo_match = echo_guard.check(text)
        if echo_match is not None:
            logger.info(
                "Dropping likely preview OpenAI Realtime echo transcript reason={} score={:.2f} transcript={!r} agent_text={!r}",
                echo_match.reason,
                echo_match.score,
                echo_match.transcript,
                echo_match.agent_text,
            )
            return
        if not allow_voice_interruptions() and active_pipeline and not active_pipeline.done():
            logger.info("Ignoring preview OpenAI Realtime transcript while interruptions are disabled: {!r}", text)
            return
        await schedule_pipeline(text, stt_final_pc=time.perf_counter())

    async def on_realtime_speech_started() -> None:
        await cancel_active_turn(notify_client=True)

    try:
        raw = await websocket.receive_text()
        start_msg = json.loads(raw)
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

        state0: dict[str, Any] | None = None
        async with get_async_session_factory()() as db:
            try:
                _, bot = await _get_active_preview_share(token, db, require_capacity=False)
            except HTTPException as exc:
                await _send_json(websocket, {"type": "error", "message": str(exc.detail)})
                await websocket.close(code=1008)
                return
            lead = await db.get(LeadRow, uuid.UUID(lead_id))
            if lead is None or lead.bot_id != bot.id:
                await _send_json(websocket, {"type": "error", "message": "Preview session not found"})
                await websocket.close(code=4404)
                return
            bid = bot.id
            bot_cfg_merged = _merge_voice_credentials(bot.to_config_dict(), settings)
            openai_key = (bot_cfg_merged.get("openai_api_key") or "").strip()
            if not openai_key:
                await _send_json(websocket, {"type": "error", "message": "OpenAI API key required for Realtime voice."})
                await websocket.close(code=4400)
                return
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

        assert bot_cfg_merged is not None
        bridge = OpenAIRealtimeVoiceBridge(
            api_key=str(bot_cfg_merged.get("openai_api_key") or ""),
            model=str(bot_cfg_merged.get("openai_realtime_model") or settings.openai_realtime_model),
            voice=str(bot_cfg_merged.get("openai_realtime_voice") or settings.openai_realtime_voice),
            transcription_model=settings.openai_realtime_transcription_model,
            instructions=bot_cfg_merged.get("openai_realtime_instructions"),
            send_json=lambda payload: _send_json(websocket, payload),
            on_transcript_final=on_realtime_transcript_final,
            on_speech_started=on_realtime_speech_started,
            allow_interruptions=allow_voice_interruptions(),
            vad_threshold=settings.openai_realtime_vad_threshold,
            vad_silence_duration_ms=settings.openai_realtime_vad_silence_duration_ms,
            vad_prefix_padding_ms=settings.openai_realtime_vad_prefix_padding_ms,
            log_context=f"route=preview_openai_realtime token={token[:8]} conversation_id={conversation_id}",
        )
        await bridge.connect()

        if state0 is not None:
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
            if reply:
                active_pipeline = asyncio.create_task(synthesize_realtime_text(reply, lambda: None))

        await _send_json(
            websocket,
            {
                "type": "ready",
                "conversation_id": conversation_id,
                "allow_interruptions": allow_voice_interruptions(),
            },
        )

        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            if message.get("type") != "websocket.receive":
                break
            if "bytes" in message and message["bytes"] is not None:
                await bridge.append_audio(message["bytes"])
                continue
            if "text" in message and message["text"] is not None:
                try:
                    ctrl = json.loads(message["text"])
                except json.JSONDecodeError:
                    continue
                ctype = ctrl.get("type")
                if ctype == "interrupt":
                    if allow_voice_interruptions():
                        await cancel_active_turn(notify_client=True)
                elif ctype == "ping":
                    await _send_json(websocket, {"type": "pong"})
    except WebSocketDisconnect:
        pass
    except asyncio.CancelledError:
        pass
    except Exception:
        try:
            await _send_json(websocket, {"type": "error", "message": "Internal server error"})
        except Exception:
            pass
    finally:
        if active_pipeline and not active_pipeline.done():
            active_pipeline.cancel()
            try:
                await active_pipeline
            except asyncio.CancelledError:
                pass
        if bridge is not None:
            await bridge.close()


@router.websocket("/api/public/agent-previews/{token}/voice-session/openai-realtime-elevenlabs")
async def public_agent_preview_openai_realtime_elevenlabs_voice_session(websocket: WebSocket, token: str) -> None:
    await websocket.accept()
    settings = get_settings()
    conversation_id: str | None = None
    bot_cfg_merged: dict[str, Any] | None = None
    bid: uuid.UUID | None = None
    generation = 0
    active_pipeline: asyncio.Task[None] | None = None
    active_greeting: asyncio.Task[None] | None = None
    bridge: OpenAIRealtimeVoiceBridge | None = None
    echo_guard = RealtimeEchoGuard()
    httpx_client = httpx.AsyncClient()

    from ai_sdr_agent.services.tool_context import voice_interruptions_allowed

    def allow_voice_interruptions() -> bool:
        return voice_interruptions_allowed(
            bool((bot_cfg_merged or {}).get("allow_voice_interruptions", True))
        )

    def invalidate_turns() -> int:
        nonlocal generation
        generation += 1
        return generation

    async def cancel_active_turn(*, notify_client: bool = True) -> None:
        invalidate_turns()
        for task in (active_pipeline, active_greeting):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        if notify_client:
            await _send_json(websocket, {"type": "agent.interrupted"})

    def elevenlabs_config() -> tuple[str, str, str, int]:
        assert bot_cfg_merged is not None
        return (
            str(bot_cfg_merged.get("elevenlabs_api_key") or ""),
            str(bot_cfg_merged.get("elevenlabs_voice_id") or ""),
            str(bot_cfg_merged.get("elevenlabs_model_id") or "eleven_turbo_v2"),
            int(get_settings().elevenlabs_optimize_streaming_latency),
        )

    async def synthesize_elevenlabs_text(
        spoken_text: str,
        mark_first_audio: Any,
        *,
        active_generation: int,
    ) -> bool:
        eleven_key, voice_id, model_id, tts_latency_opt = elevenlabs_config()
        echo_guard.record_agent_speech(spoken_text)
        return await stream_elevenlabs_text_to_ws(
            spoken_text,
            httpx_client=httpx_client,
            elevenlabs_api_key=eleven_key,
            voice_id=voice_id,
            model_id=model_id,
            optimize_streaming_latency=tts_latency_opt,
            send_json=lambda payload: _send_json(websocket, payload),
            should_continue=lambda: active_generation == generation,
            mark_first_audio=mark_first_audio,
        )

    async def pipeline(user_text: str, active_generation: int, *, stt_final_pc: float) -> None:
        nonlocal conversation_id, bot_cfg_merged, bid
        assert conversation_id and bot_cfg_merged and bid
        eleven_key, voice_id, _, _ = elevenlabs_config()
        await run_voice_graph_turn(
            bot_id=str(bid),
            bid=bid,
            bot_cfg=bot_cfg_merged,
            conversation_id=conversation_id,
            user_text=user_text,
            stt_final_pc=stt_final_pc,
            send_json=lambda payload: _send_json(websocket, payload),
            synthesize_text=lambda spoken_text, mark_first_audio: synthesize_elevenlabs_text(
                spoken_text,
                mark_first_audio,
                active_generation=active_generation,
            ),
            has_speech_output=lambda: bool(eleven_key and voice_id),
            is_current=lambda: active_generation == generation,
        )

    async def schedule_pipeline(user_text: str, *, stt_final_pc: float) -> None:
        nonlocal active_pipeline
        await cancel_active_turn(notify_client=False)
        active_generation = invalidate_turns()
        active_pipeline = asyncio.create_task(pipeline(user_text, active_generation, stt_final_pc=stt_final_pc))

    async def on_realtime_transcript_final(text: str) -> None:
        echo_match = echo_guard.check(text)
        if echo_match is not None:
            logger.info(
                "Dropping likely preview OpenAI Realtime hybrid echo transcript reason={} score={:.2f} transcript={!r} agent_text={!r}",
                echo_match.reason,
                echo_match.score,
                echo_match.transcript,
                echo_match.agent_text,
            )
            return
        if not allow_voice_interruptions() and (
            (active_pipeline and not active_pipeline.done()) or (active_greeting and not active_greeting.done())
        ):
            logger.info(
                "Ignoring preview OpenAI Realtime hybrid transcript while interruptions are disabled: {!r}",
                text,
            )
            return
        await schedule_pipeline(text, stt_final_pc=time.perf_counter())

    async def on_realtime_speech_started() -> None:
        await cancel_active_turn(notify_client=True)

    async def send_initial_greeting(state0: dict[str, Any], active_generation: int) -> None:
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
        ok = await synthesize_elevenlabs_text(reply, lambda: None, active_generation=active_generation)
        if ok and active_generation == generation:
            await _send_json(websocket, {"type": "agent.done"})

    try:
        raw = await websocket.receive_text()
        start_msg = json.loads(raw)
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

        state0: dict[str, Any] | None = None
        async with get_async_session_factory()() as db:
            try:
                _, bot = await _get_active_preview_share(token, db, require_capacity=False)
            except HTTPException as exc:
                await _send_json(websocket, {"type": "error", "message": str(exc.detail)})
                await websocket.close(code=1008)
                return
            lead = await db.get(LeadRow, uuid.UUID(lead_id))
            if lead is None or lead.bot_id != bot.id:
                await _send_json(websocket, {"type": "error", "message": "Preview session not found"})
                await websocket.close(code=4404)
                return
            bid = bot.id
            bot_cfg_merged = _merge_voice_credentials(bot.to_config_dict(), settings)
            openai_key = (bot_cfg_merged.get("openai_api_key") or "").strip()
            eleven_key = (bot_cfg_merged.get("elevenlabs_api_key") or "").strip()
            voice_id = (bot_cfg_merged.get("elevenlabs_voice_id") or "").strip()
            if not openai_key:
                await _send_json(websocket, {"type": "error", "message": "OpenAI API key required for hybrid voice."})
                await websocket.close(code=4400)
                return
            if not eleven_key or not voice_id:
                await _send_json(
                    websocket,
                    {"type": "error", "message": "ElevenLabs API key and voice id required for hybrid voice."},
                )
                await websocket.close(code=4400)
                return

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

        assert bot_cfg_merged is not None
        bridge = OpenAIRealtimeVoiceBridge(
            api_key=str(bot_cfg_merged.get("openai_api_key") or ""),
            model=str(bot_cfg_merged.get("openai_realtime_model") or settings.openai_realtime_model),
            voice=str(bot_cfg_merged.get("openai_realtime_voice") or settings.openai_realtime_voice),
            transcription_model=settings.openai_realtime_transcription_model,
            instructions=bot_cfg_merged.get("openai_realtime_instructions"),
            send_json=lambda payload: _send_json(websocket, payload),
            on_transcript_final=on_realtime_transcript_final,
            on_speech_started=on_realtime_speech_started,
            enable_audio_output=False,
            allow_interruptions=allow_voice_interruptions(),
            vad_threshold=settings.openai_realtime_vad_threshold,
            vad_silence_duration_ms=settings.openai_realtime_vad_silence_duration_ms,
            vad_prefix_padding_ms=settings.openai_realtime_vad_prefix_padding_ms,
            log_context=f"route=preview_openai_realtime_elevenlabs token={token[:8]} conversation_id={conversation_id}",
        )
        await bridge.connect()

        if state0 is not None:
            active_generation = invalidate_turns()
            active_greeting = asyncio.create_task(send_initial_greeting(state0, active_generation))

        await _send_json(
            websocket,
            {
                "type": "ready",
                "conversation_id": conversation_id,
                "allow_interruptions": allow_voice_interruptions(),
            },
        )

        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            if message.get("type") != "websocket.receive":
                break
            if "bytes" in message and message["bytes"] is not None:
                await bridge.append_audio(message["bytes"])
                continue
            if "text" in message and message["text"] is not None:
                try:
                    ctrl = json.loads(message["text"])
                except json.JSONDecodeError:
                    continue
                ctype = ctrl.get("type")
                if ctype == "interrupt":
                    if allow_voice_interruptions():
                        await cancel_active_turn(notify_client=True)
                elif ctype == "ping":
                    await _send_json(websocket, {"type": "pong"})
    except WebSocketDisconnect:
        pass
    except asyncio.CancelledError:
        pass
    except Exception:
        try:
            await _send_json(websocket, {"type": "error", "message": "Internal server error"})
        except Exception:
            pass
    finally:
        for task in (active_pipeline, active_greeting):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        if bridge is not None:
            await bridge.close()
        await httpx_client.aclose()
