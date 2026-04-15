import asyncio
from collections import deque

import pytest

from ai_sdr_agent.vocode_agent import SDRVocodeAgent, build_agent_config


class RecordingConversationService:
    def __init__(self):
        self.start_session_calls: list[tuple[str, str | None]] = []
        self.handle_turn_calls: list[str] = []
        self.started_inputs: asyncio.Queue[str] = asyncio.Queue()
        self._turn_blockers: deque[asyncio.Event] = deque()
        self._turn_count = 0

    def block_next_turn(self) -> asyncio.Event:
        blocker = asyncio.Event()
        self._turn_blockers.append(blocker)
        return blocker

    async def start_session(
        self,
        lead_id: str,
        *,
        conversation_id: str | None = None,
        bot_config: dict | None = None,
    ) -> str:
        self.start_session_calls.append((lead_id, conversation_id))
        return conversation_id or "conv-test"

    async def handle_turn(self, conversation_id: str, human_input: str) -> dict:
        self.handle_turn_calls.append(human_input)
        self.started_inputs.put_nowait(human_input)
        self._turn_count += 1
        if self._turn_blockers:
            blocker = self._turn_blockers.popleft()
            await blocker.wait()
        return {
            "last_agent_response": f"reply:{human_input}",
            "turn_count": self._turn_count,
            "route_decision": "qualify_lead",
        }


@pytest.mark.asyncio
async def test_agent_coalesces_interrupt_refinements():
    service = RecordingConversationService()
    release_first_turn = service.block_next_turn()
    agent = SDRVocodeAgent(
        agent_config=build_agent_config(
            lead_id="lead-001",
            calendar_id="sales-team",
            sales_rep_name="Taylor Morgan",
            initial_message_text="Hi there",
        ),
        conversation_service=service,
    )
    conversation_id = "conv-test"

    first_turn = asyncio.create_task(
        agent.respond("yes", conversation_id, is_interrupt=False)
    )
    assert await service.started_inputs.get() == "yes"

    stale_interrupt = asyncio.create_task(
        agent.respond("um", conversation_id, is_interrupt=True)
    )
    latest_interrupt = asyncio.create_task(
        agent.respond("um we have", conversation_id, is_interrupt=True)
    )

    await asyncio.sleep(0)
    release_first_turn.set()

    first_result, stale_result, latest_result = await asyncio.gather(
        first_turn,
        stale_interrupt,
        latest_interrupt,
    )

    assert first_result == ("reply:yes", False)
    assert stale_result == (None, False)
    assert latest_result == ("reply:um we have", False)
    assert service.handle_turn_calls == ["yes", "um we have"]
    assert len(service.start_session_calls) == 1


@pytest.mark.asyncio
async def test_agent_keeps_turn_worker_alive_after_caller_cancellation():
    service = RecordingConversationService()
    release_first_turn = service.block_next_turn()
    agent = SDRVocodeAgent(
        agent_config=build_agent_config(
            lead_id="lead-001",
            calendar_id="sales-team",
            sales_rep_name="Taylor Morgan",
            initial_message_text="Hi there",
        ),
        conversation_service=service,
    )
    conversation_id = "conv-test"

    active_interrupt = asyncio.create_task(
        agent.respond("um", conversation_id, is_interrupt=True)
    )
    assert await service.started_inputs.get() == "um"

    active_interrupt.cancel()
    with pytest.raises(asyncio.CancelledError):
        await active_interrupt

    duplicate_result = await agent.respond("um", conversation_id, is_interrupt=True)
    latest_interrupt = asyncio.create_task(
        agent.respond("um we have", conversation_id, is_interrupt=True)
    )

    await asyncio.sleep(0)
    assert service.handle_turn_calls == ["um"]

    release_first_turn.set()
    latest_result = await latest_interrupt

    assert duplicate_result == (None, False)
    assert latest_result == ("reply:um we have", False)
    assert service.handle_turn_calls == ["um", "um we have"]
    assert len(service.start_session_calls) == 1
