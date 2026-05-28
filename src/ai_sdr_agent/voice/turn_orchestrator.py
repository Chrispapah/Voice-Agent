from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Awaitable, Callable

from loguru import logger

from ai_sdr_agent.db.engine import get_async_session_factory
from ai_sdr_agent.db.repositories import PgCallLogRepository, PgLeadRepository, PgSessionStore
from ai_sdr_agent.routers.test_sessions import _build_service_for_bot
from ai_sdr_agent.services.latency_analytics import WebVoiceTurnSample, shared_latency_analytics
from ai_sdr_agent.services.tool_context import reset_tool_sound_callback, set_tool_sound_callback
from ai_sdr_agent.text.tts_sentence_buffer import SentenceStreamBuffer

SendJson = Callable[[dict[str, Any]], Awaitable[None]]
IsCurrent = Callable[[], bool]
MarkFirstAudio = Callable[[], None]
SynthesizeText = Callable[[str, MarkFirstAudio], Awaitable[bool]]
HasSpeechOutput = Callable[[], bool]


async def run_voice_graph_turn(
    *,
    bot_id: str,
    bid: uuid.UUID,
    bot_cfg: dict[str, Any],
    conversation_id: str,
    user_text: str,
    stt_final_pc: float,
    send_json: SendJson,
    synthesize_text: SynthesizeText,
    has_speech_output: HasSpeechOutput,
    is_current: IsCurrent,
) -> None:
    """Run one finalized voice transcript through the existing LangGraph service.

    Transport adapters provide STT and TTS. This function owns the text turn
    boundary so graph/subagent/tool/KB behavior stays identical across providers.
    """
    if not user_text.strip():
        return

    pipeline_enter_pc = time.perf_counter()
    first_llm_pc: list[float | None] = [None]
    first_phrase_pc: list[float | None] = [None]
    first_audio_pc: list[float | None] = [None]
    graph_done_pc: list[float | None] = [None]
    streamed_llm = False

    def _pc_delta_ms(start: float, end: float | None) -> float | None:
        if end is None:
            return None
        return (end - start) * 1000.0

    def _mark_first_audio() -> None:
        if first_audio_pc[0] is None:
            first_audio_pc[0] = time.perf_counter()

    async def _record_metrics() -> None:
        if not is_current():
            return
        turn_end_pc = time.perf_counter()
        fp, fa = first_phrase_pc[0], first_audio_pc[0]
        phrase_to_audio_ms = (fa - fp) * 1000.0 if fp is not None and fa is not None else None
        sample = WebVoiceTurnSample(
            conversation_id=conversation_id,
            bot_id=str(bot_id),
            streamed_llm=streamed_llm,
            stt_final_to_pipeline_ms=(pipeline_enter_pc - stt_final_pc) * 1000.0,
            pipeline_to_first_llm_token_ms=_pc_delta_ms(pipeline_enter_pc, first_llm_pc[0]),
            pipeline_to_first_phrase_ms=_pc_delta_ms(pipeline_enter_pc, first_phrase_pc[0]),
            pipeline_to_first_tts_byte_ms=_pc_delta_ms(pipeline_enter_pc, first_audio_pc[0]),
            first_phrase_to_first_tts_byte_ms=phrase_to_audio_ms,
            pipeline_to_graph_done_ms=_pc_delta_ms(pipeline_enter_pc, graph_done_pc[0]),
            pipeline_to_turn_end_ms=(turn_end_pc - pipeline_enter_pc) * 1000.0,
            stt_final_to_first_tts_byte_ms=_pc_delta_ms(stt_final_pc, first_audio_pc[0]),
            recorded_at=time.time(),
        )
        logger.info(
            "web_voice_latency conversation_id={} bot_id={} streamed_llm={} "
            "stt_final_to_pipeline_ms={:.0f} pipeline_to_first_llm_token_ms={} pipeline_to_first_phrase_ms={} "
            "pipeline_to_first_tts_byte_ms={} first_phrase_to_first_tts_byte_ms={} "
            "pipeline_to_graph_done_ms={} stt_final_to_first_tts_byte_ms={} pipeline_to_turn_end_ms={:.0f}",
            sample.conversation_id,
            sample.bot_id,
            sample.streamed_llm,
            sample.stt_final_to_pipeline_ms,
            f"{sample.pipeline_to_first_llm_token_ms:.0f}"
            if sample.pipeline_to_first_llm_token_ms is not None
            else "n/a",
            f"{sample.pipeline_to_first_phrase_ms:.0f}" if sample.pipeline_to_first_phrase_ms is not None else "n/a",
            f"{sample.pipeline_to_first_tts_byte_ms:.0f}"
            if sample.pipeline_to_first_tts_byte_ms is not None
            else "n/a",
            f"{sample.first_phrase_to_first_tts_byte_ms:.0f}"
            if sample.first_phrase_to_first_tts_byte_ms is not None
            else "n/a",
            f"{sample.pipeline_to_graph_done_ms:.0f}" if sample.pipeline_to_graph_done_ms is not None else "n/a",
            f"{sample.stt_final_to_first_tts_byte_ms:.0f}"
            if sample.stt_final_to_first_tts_byte_ms is not None
            else "n/a",
            sample.pipeline_to_turn_end_ms,
        )
        await shared_latency_analytics.record_web_voice_turn(sample)

    async def _tool_sound(payload: dict[str, Any]) -> None:
        await send_json({"type": "tool.sound", **payload})

    sound_token = set_tool_sound_callback(_tool_sound)
    try:
        async with get_async_session_factory()() as db:
            lead_repo = PgLeadRepository(db)
            session_store = PgSessionStore(db, bid)
            call_log_repo = PgCallLogRepository(db, bid)
            svc = _build_service_for_bot(bot_cfg, lead_repo, session_store, call_log_repo)
            brain = svc.dependencies.brain

            if brain.supports_response_token_stream():
                streamed_llm = True
                phrase_parts: list[str] = []
                tts_streams_started = False

                async def _on_phrase(phrase: str) -> None:
                    nonlocal tts_streams_started
                    raw = phrase.strip()
                    if not raw or not is_current():
                        return
                    if first_phrase_pc[0] is None:
                        first_phrase_pc[0] = time.perf_counter()
                    phrase_parts.append(raw)
                    cumulative = " ".join(phrase_parts).strip()
                    if not has_speech_output():
                        await send_json({"type": "agent.text", "text": cumulative})
                        return
                    tts_streams_started = True
                    _, ok_audio = await asyncio.gather(
                        send_json({"type": "agent.text", "text": cumulative}),
                        synthesize_text(raw, _mark_first_audio),
                    )
                    if not ok_audio:
                        return

                streamed = await svc.start_streamed_turn(conversation_id, user_text)
                buf = SentenceStreamBuffer()
                async for token in streamed.chunks:
                    if first_llm_pc[0] is None and token:
                        first_llm_pc[0] = time.perf_counter()
                    if not is_current():
                        break
                    await buf.feed(_on_phrase, token)
                await buf.flush(_on_phrase)
                state = await streamed.final_state_task
                await db.commit()
                graph_done_pc[0] = time.perf_counter()
                if not is_current():
                    return
                reply = (state.get("last_agent_response") or "").strip()
                if reply and not tts_streams_started and has_speech_output():
                    await asyncio.gather(
                        send_json({"type": "agent.text", "text": reply}),
                        synthesize_text(reply, _mark_first_audio),
                    )
                await send_json(
                    {
                        "type": "agent.text",
                        "text": reply,
                        "stage": state.get("current_node"),
                        "next_node": state.get("next_node"),
                    }
                )
                if not has_speech_output():
                    await send_json({"type": "error", "message": "Speech output is not configured."})
            else:
                state = await svc.handle_turn(conversation_id, user_text)
                await db.commit()
                graph_done_pc[0] = time.perf_counter()
                if not is_current():
                    return
                reply = (state.get("last_agent_response") or "").strip()
                await send_json(
                    {
                        "type": "agent.text",
                        "text": reply,
                        "stage": state.get("current_node"),
                        "next_node": state.get("next_node"),
                    }
                )
                if not reply or not is_current():
                    await send_json({"type": "agent.done"})
                    if is_current():
                        await _record_metrics()
                    return
                if not has_speech_output():
                    await send_json({"type": "error", "message": "Speech output is not configured."})
                    await send_json({"type": "agent.done"})
                    if is_current():
                        await _record_metrics()
                    return
                await synthesize_text(reply, _mark_first_audio)

        if is_current():
            await _record_metrics()
        if is_current():
            await send_json({"type": "agent.done"})
    except asyncio.CancelledError:
        await send_json({"type": "agent.interrupted"})
        raise
    finally:
        reset_tool_sound_callback(sound_token)
