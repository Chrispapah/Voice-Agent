from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, WebSocket
from jose import JWTError
from loguru import logger
from starlette.websockets import WebSocketDisconnect

from ai_sdr_agent.auth.dependencies import decode_supabase_jwt
from ai_sdr_agent.config import get_settings
from ai_sdr_agent.db.engine import get_async_session_factory
from ai_sdr_agent.db.repositories import PgCallLogRepository, PgLeadRepository, PgSessionStore
from ai_sdr_agent.routers.test_sessions import _build_service_for_bot, _verify_bot
from ai_sdr_agent.routers.web_voice import _merge_voice_credentials, _send_json
from ai_sdr_agent.voice.echo_filter import RealtimeEchoGuard
from ai_sdr_agent.voice.openai_realtime import OpenAIRealtimeVoiceBridge
from ai_sdr_agent.voice.turn_orchestrator import run_voice_graph_turn

router = APIRouter(prefix="/api/bots", tags=["voice"])


@router.websocket("/{bot_id}/voice-session/openai-realtime")
async def openai_realtime_voice_session(websocket: WebSocket, bot_id: str) -> None:
    await websocket.accept()
    settings = get_settings()
    user_id: uuid.UUID | None = None
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
            bot_id=bot_id,
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
                "Dropping likely OpenAI Realtime echo transcript reason={} score={:.2f} transcript={!r} agent_text={!r}",
                echo_match.reason,
                echo_match.score,
                echo_match.transcript,
                echo_match.agent_text,
            )
            return
        if not allow_voice_interruptions() and active_pipeline and not active_pipeline.done():
            logger.info("Ignoring OpenAI Realtime transcript while interruptions are disabled: {!r}", text)
            return
        await schedule_pipeline(text, stt_final_pc=time.perf_counter())

    async def on_realtime_speech_started() -> None:
        await cancel_active_turn(notify_client=True)

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

        state0: dict[str, Any] | None = None
        async with get_async_session_factory()() as db:
            try:
                bot = await _verify_bot(bot_id, user_id, db)
            except HTTPException as exc:
                await _send_json(websocket, {"type": "error", "message": str(exc.detail)})
                await websocket.close(code=1008)
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
            log_context=f"route=openai_realtime bot_id={bot_id} conversation_id={conversation_id}",
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
        logger.exception("openai_realtime_voice_session handler error")
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
