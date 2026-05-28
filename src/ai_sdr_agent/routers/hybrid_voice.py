from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any

import httpx
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
from ai_sdr_agent.voice.elevenlabs_tts import stream_elevenlabs_text_to_ws
from ai_sdr_agent.voice.echo_filter import RealtimeEchoGuard
from ai_sdr_agent.voice.openai_realtime import OpenAIRealtimeVoiceBridge
from ai_sdr_agent.voice.turn_orchestrator import run_voice_graph_turn

router = APIRouter(prefix="/api/bots", tags=["voice"])


@router.websocket("/{bot_id}/voice-session/openai-realtime-elevenlabs")
async def openai_realtime_elevenlabs_voice_session(websocket: WebSocket, bot_id: str) -> None:
    await websocket.accept()
    settings = get_settings()
    user_id: uuid.UUID | None = None
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
            bot_id=bot_id,
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
                "Dropping likely OpenAI Realtime hybrid echo transcript reason={} score={:.2f} transcript={!r} agent_text={!r}",
                echo_match.reason,
                echo_match.score,
                echo_match.transcript,
                echo_match.agent_text,
            )
            return
        if not allow_voice_interruptions() and (
            (active_pipeline and not active_pipeline.done()) or (active_greeting and not active_greeting.done())
        ):
            logger.info("Ignoring OpenAI Realtime hybrid transcript while interruptions are disabled: {!r}", text)
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
            log_context=f"route=openai_realtime_elevenlabs bot_id={bot_id} conversation_id={conversation_id}",
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
        logger.exception("openai_realtime_elevenlabs_voice_session handler error")
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
