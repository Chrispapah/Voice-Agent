import pytest

from ai_sdr_agent.services.latency_analytics import LatencyAnalyticsBuffer, WebVoiceTurnSample


@pytest.mark.asyncio
async def test_latency_buffer_records_and_summarizes():
    buf = LatencyAnalyticsBuffer(maxlen=100)
    for i in range(5):
        await buf.record_turn(
            conversation_id=f"c-{i}",
            turn_count=i + 1,
            route_decision="qualify_lead",
            latency_total_ms=100.0 + i,
            latency_graph_ms=80.0,
            latency_persist_ms=20.0,
        )
    snap = await buf.snapshot(recent_limit=10)
    assert snap["sample_count"] == 5
    assert snap["latency_total_ms"]["count"] == 5
    assert snap["latency_total_ms"]["min_ms"] == 100.0
    assert snap["latency_total_ms"]["max_ms"] == 104.0
    assert snap["by_route_decision"]["qualify_lead"]["count"] == 5
    assert len(snap["recent_turns"]) == 5


@pytest.mark.asyncio
async def test_empty_buffer_snapshot():
    buf = LatencyAnalyticsBuffer(maxlen=10)
    snap = await buf.snapshot()
    assert snap["sample_count"] == 0
    assert snap["latency_total_ms"]["mean_ms"] is None
    assert snap["web_voice"]["sample_count"] == 0


@pytest.mark.asyncio
async def test_web_voice_samples_in_snapshot():
    buf = LatencyAnalyticsBuffer(maxlen=50)
    await buf.record_web_voice_turn(
        WebVoiceTurnSample(
            conversation_id="c1",
            bot_id="b1",
            streamed_llm=True,
            stt_final_to_pipeline_ms=5.0,
            pipeline_to_first_llm_token_ms=100.0,
            pipeline_to_first_phrase_ms=120.0,
            pipeline_to_first_tts_byte_ms=250.0,
            first_phrase_to_first_tts_byte_ms=130.0,
            pipeline_to_graph_done_ms=400.0,
            pipeline_to_turn_end_ms=450.0,
            stt_final_to_first_tts_byte_ms=255.0,
            recorded_at=0.0,
        )
    )
    snap = await buf.snapshot()
    assert snap["web_voice"]["sample_count"] == 1
    assert snap["web_voice"]["stt_final_to_pipeline_ms"]["mean_ms"] == 5.0
    assert snap["web_voice"]["recent"][0]["conversation_id"] == "c1"
