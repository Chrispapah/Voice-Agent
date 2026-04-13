import pytest

from ai_sdr_agent.services.latency_analytics import LatencyAnalyticsBuffer


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
