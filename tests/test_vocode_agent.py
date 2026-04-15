"""Tests for SDRVocodeAgent parallel acknowledgment prefill."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from vocode.streaming.agent.base_agent import AgentResponseMessage
from vocode.streaming.models.actions import EndOfTurn
from vocode.streaming.models.message import BotBackchannel

from ai_sdr_agent.config import parse_agent_prefill_ack_phrases
from ai_sdr_agent.vocode_agent import SDRAgentConfig, SDRVocodeAgent


def _minimal_state(next_node: str = "qualify") -> dict:
    return {"next_node": next_node, "last_agent_response": "Thanks for sharing."}


def _make_agent(
    *,
    prefill_ack_enabled: bool = True,
    prefill_ack_phrases: tuple[str, ...] = ("Okay.", "Got it."),
) -> tuple[SDRVocodeAgent, AsyncMock]:
    cfg = SDRAgentConfig(
        lead_id="lead-1",
        calendar_id="cal",
        sales_rep_name="Rep",
        initial_message_text="Hi",
        prefill_ack_enabled=prefill_ack_enabled,
        prefill_ack_phrases=prefill_ack_phrases,
        generate_responses=False,
    )
    svc = AsyncMock()
    svc.start_session = AsyncMock()
    svc.get_state = AsyncMock(return_value=_minimal_state("qualify"))
    svc.handle_turn = AsyncMock(
        return_value={**_minimal_state(), "last_agent_response": "Main reply."}
    )
    agent = SDRVocodeAgent(agent_config=cfg, conversation_service=svc)
    return agent, svc


@pytest.mark.asyncio
async def test_prefill_emits_backchannel_before_handle_turn():
    agent, svc = _make_agent()
    sequence: list[str] = []

    async def track_get_state(cid: str):
        sequence.append("get_state")
        return _minimal_state("qualify")

    async def track_handle_turn(cid: str, text: str):
        sequence.append("handle_turn")
        return {**_minimal_state(), "last_agent_response": "Main reply."}

    svc.get_state = track_get_state
    svc.handle_turn = track_handle_turn

    orig_produce = agent.produce_interruptible_agent_response_event_nonblocking

    def tracking_produce(*args, **kwargs):
        sequence.append("produce")
        return orig_produce(*args, **kwargs)

    with patch.object(
        agent,
        "produce_interruptible_agent_response_event_nonblocking",
        side_effect=tracking_produce,
    ) as produce:
        await agent.respond("Hello there", "conv-1", is_interrupt=False)

    assert sequence == ["get_state", "produce", "handle_turn"]
    produce_calls = produce.call_args_list
    assert len(produce_calls) >= 1
    first_msg = produce_calls[0][0][0]
    assert isinstance(first_msg, AgentResponseMessage)
    assert isinstance(first_msg.message, BotBackchannel)
    assert first_msg.message.text in ("Okay.", "Got it.")
    svc.start_session.assert_awaited_once()
    assert produce.called


@pytest.mark.asyncio
async def test_prefill_skipped_when_disabled():
    agent, svc = _make_agent(prefill_ack_enabled=False)
    with patch.object(agent, "produce_interruptible_agent_response_event_nonblocking") as produce:
        await agent.respond("Hello", "conv-2", is_interrupt=False)
    produce.assert_not_called()
    svc.handle_turn.assert_awaited()


@pytest.mark.asyncio
async def test_prefill_skipped_empty_human_input():
    agent, svc = _make_agent()
    with patch.object(agent, "produce_interruptible_agent_response_event_nonblocking") as produce:
        await agent.respond("   ", "conv-3", is_interrupt=False)
    produce.assert_not_called()


@pytest.mark.asyncio
async def test_prefill_skipped_on_interrupt():
    agent, svc = _make_agent()
    with patch.object(agent, "produce_interruptible_agent_response_event_nonblocking") as produce:
        await agent.respond("Hello", "conv-4", is_interrupt=True)
    produce.assert_not_called()


@pytest.mark.asyncio
async def test_prefill_skipped_when_already_complete():
    agent, svc = _make_agent()
    svc.get_state = AsyncMock(return_value=_minimal_state("complete"))
    with patch.object(agent, "produce_interruptible_agent_response_event_nonblocking") as produce:
        await agent.respond("Hello", "conv-5", is_interrupt=False)
    produce.assert_not_called()


@pytest.mark.asyncio
async def test_prefill_emitted_with_empty_response_returns_fallback_text():
    agent, svc = _make_agent()
    svc.get_state = AsyncMock(return_value=_minimal_state("qualify"))
    svc.handle_turn = AsyncMock(
        return_value={
            **_minimal_state("complete"),
            "last_agent_response": "",
        }
    )
    with patch.object(agent, "produce_interruptible_agent_response_event_nonblocking"):
        response, should_stop = await agent.respond("Hello", "conv-6", is_interrupt=False)
    assert response == "Thanks for your time. Goodbye."
    assert should_stop is True


@pytest.mark.asyncio
async def test_prefill_ack_is_always_interruptible():
    agent, _ = _make_agent()
    with patch.object(agent, "produce_interruptible_agent_response_event_nonblocking") as produce:
        await agent.respond("Hello", "conv-7", is_interrupt=False)
    assert produce.call_count == 1
    assert produce.call_args.kwargs["is_interruptible"] is True


@pytest.mark.asyncio
async def test_concurrent_respond_calls_are_serialized_per_conversation():
    agent, svc = _make_agent()
    started = asyncio.Event()
    release = asyncio.Event()
    call_order: list[str] = []

    async def slow_handle_turn(cid: str, text: str):
        call_order.append(text)
        if text == "first":
            started.set()
            await release.wait()
        return {**_minimal_state(), "last_agent_response": f"reply:{text}"}

    svc.handle_turn = AsyncMock(side_effect=slow_handle_turn)

    first_task = asyncio.create_task(agent.respond("first", "conv-serial", is_interrupt=False))
    await started.wait()
    second_task = asyncio.create_task(agent.respond("second", "conv-serial", is_interrupt=False))
    await asyncio.sleep(0)

    assert svc.handle_turn.await_count == 1

    release.set()
    first_result = await first_task
    second_result = await second_task

    assert first_result == ("reply:first", False)
    assert second_result == ("reply:second", False)
    assert call_order == ["first", "second"]
    svc.start_session.assert_awaited_once()


def test_parse_agent_prefill_ack_phrases_pipe_separated():
    assert parse_agent_prefill_ack_phrases("A.|B.") == ("A.", "B.")


def test_parse_agent_prefill_ack_phrases_empty_falls_back_to_defaults():
    out = parse_agent_prefill_ack_phrases("   ")
    assert len(out) >= 1
    assert "Okay." in out or "Got it." in out
