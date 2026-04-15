import asyncio

import pytest

from ai_sdr_agent.vocode_agent import SDRVocodeAgent, build_agent_config


class RecordingConversationService:
    def __init__(self):
        self.start_session_calls: list[tuple[str, str | None]] = []
        self.handle_turn_calls: list[str] = []
        self._first_turn_started = asyncio.Event()
        self._release_first_turn = asyncio.Event()
        self._turn_count = 0

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
        self._turn_count += 1
        if self._turn_count == 1:
            self._first_turn_started.set()
            await self._release_first_turn.wait()
        return {
            "last_agent_response": f"reply:{human_input}",
            "turn_count": self._turn_count,
            "route_decision": "qualify_lead",
        }


@pytest.mark.asyncio
async def test_agent_coalesces_interrupt_refinements():
    service = RecordingConversationService()
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
    await service._first_turn_started.wait()

    stale_interrupt = asyncio.create_task(
        agent.respond("um", conversation_id, is_interrupt=True)
    )
    latest_interrupt = asyncio.create_task(
        agent.respond("um we have", conversation_id, is_interrupt=True)
    )

    await asyncio.sleep(0)
    service._release_first_turn.set()

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
