import asyncio
from copy import deepcopy
from pathlib import Path
import time

import pytest

from ai_sdr_agent.graph.service import SDRConversationService, SDRRuntimeDependencies
from ai_sdr_agent.graph.spec import ConversationSpecV1, graph_execution_kind, parse_conversation_spec
from ai_sdr_agent.graph.state import _DEFAULT_BOT_CONFIG
from ai_sdr_agent.services.brain import StubConversationBrain
from ai_sdr_agent.services.persistence import (
    InMemoryCallLogRepository,
    InMemoryLeadRepository,
    InMemorySessionStore,
)
from ai_sdr_agent.services.pre_call_loader import PreCallLoader
from ai_sdr_agent.tools import StubCRMGateway, StubCalendarGateway, StubEmailGateway


def _stub_bot_config(**overrides):
    cfg = {**deepcopy(dict(_DEFAULT_BOT_CONFIG)), **overrides}
    return cfg


def _build_conversation_service(brain=None, bot_config=None):
    lead_repository = InMemoryLeadRepository()
    seed_lead = lead_repository._leads["lead-001"]
    calendar_gateway = StubCalendarGateway()
    email_gateway = StubEmailGateway()
    crm_gateway = StubCRMGateway(seed_leads=[seed_lead])
    service = SDRConversationService(
        SDRRuntimeDependencies(
            brain=brain or StubConversationBrain(),
            calendar_gateway=calendar_gateway,
            email_gateway=email_gateway,
            crm_gateway=crm_gateway,
            pre_call_loader=PreCallLoader(
                lead_repository=lead_repository,
                calendar_gateway=calendar_gateway,
            ),
            session_store=InMemorySessionStore(),
            call_log_repository=InMemoryCallLogRepository(),
            email_template_path=Path("templates/follow_up_email.html"),
            sales_rep_name="Taylor Morgan",
            from_name="AI SDR",
        ),
        bot_config=bot_config,
    )
    return service, calendar_gateway, email_gateway, crm_gateway


@pytest.fixture
def conversation_service():
    return _build_conversation_service()


class SlowStubConversationBrain(StubConversationBrain):
    async def respond(
        self,
        *,
        system_prompt: str,
        transcript: list[dict[str, str]],
        max_tokens: int | None = None,
        trace: dict | None = None,
    ) -> str:
        await asyncio.sleep(0.05)
        return await super().respond(
            system_prompt=system_prompt,
            transcript=transcript,
            max_tokens=max_tokens,
            trace=trace,
        )

    async def classify(
        self,
        *,
        instruction: str,
        human_input: str,
        labels,
        trace: dict | None = None,
    ) -> str:
        await asyncio.sleep(0.05)
        return await super().classify(
            instruction=instruction,
            human_input=human_input,
            labels=labels,
            trace=trace,
        )

    async def extract_qualification(
        self,
        *,
        transcript: list[dict[str, str]],
        existing_pain_points: list[str],
        trace: dict | None = None,
    ) -> dict:
        await asyncio.sleep(0.05)
        return await super().extract_qualification(
            transcript=transcript,
            existing_pain_points=existing_pain_points,
            trace=trace,
        )


@pytest.mark.asyncio
async def test_graph_advances_one_stage_per_turn(conversation_service):
    service, _, _, _ = conversation_service
    conversation_id = await service.start_session("lead-001")

    state = await service.handle_turn(conversation_id, "")
    assert state["current_node"] == "greeting"
    assert state["next_node"] == "qualify_lead"

    state = await service.handle_turn(conversation_id, "Yes, I oversee sales operations.")
    assert state["current_node"] == "qualify_lead"
    assert state["next_node"] == "pitch"

    state = await service.handle_turn(conversation_id, "Yes, sounds interesting.")
    assert state["current_node"] == "pitch"
    assert state["next_node"] == "book_meeting"


@pytest.mark.asyncio
async def test_booking_path_updates_state_and_side_effects(conversation_service):
    service, calendar_gateway, email_gateway, crm_gateway = conversation_service
    conversation_id = await service.start_session("lead-001")

    await service.handle_turn(conversation_id, "")
    await service.handle_turn(conversation_id, "Yes, I run sales operations.")
    await service.handle_turn(conversation_id, "Yes, let's do it.")
    state = await service.handle_turn(conversation_id, "Tomorrow at 3 PM works.")

    assert state["current_node"] == "book_meeting"
    assert state["meeting_booked"] is True
    assert state["next_node"] == "wrap_up"
    assert len(calendar_gateway.bookings) == 1

    state = await service.handle_turn(conversation_id, "Booked sounds good.")
    assert state["current_node"] == "wrap_up"
    assert state["call_outcome"] == "meeting_booked"
    assert len(email_gateway.sent_messages) == 1
    assert len(crm_gateway.updates) == 1


@pytest.mark.asyncio
async def test_objection_loops_back_to_pitch(conversation_service):
    service, _, _, _ = conversation_service
    conversation_id = await service.start_session("lead-001")

    await service.handle_turn(conversation_id, "")
    await service.handle_turn(conversation_id, "Yes, I lead sales ops.")
    state = await service.handle_turn(conversation_id, "We already have a process.")

    assert state["current_node"] == "pitch"
    assert state["next_node"] == "handle_objection"

    state = await service.handle_turn(conversation_id, "Maybe later, send info.")
    assert state["current_node"] == "handle_objection"
    assert state["next_node"] in {"pitch", "wrap_up"}


@pytest.mark.asyncio
async def test_handle_turn_serializes_same_conversation_updates():
    service, _, _, _ = _build_conversation_service(brain=SlowStubConversationBrain())
    conversation_id = await service.start_session("lead-001")

    await service.handle_turn(conversation_id, "")

    start = time.perf_counter()
    await asyncio.gather(
        service.handle_turn(conversation_id, "Yes, I lead sales ops."),
        service.handle_turn(conversation_id, "Yes, sounds interesting."),
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    state = await service.get_state(conversation_id)
    human_messages = [message for message in state["transcript"] if message["role"] == "human"]

    assert state["turn_count"] == 2
    assert len(human_messages) == 2
    assert state["current_node"] == "pitch"
    assert state["next_node"] == "book_meeting"
    assert elapsed_ms >= 80


def test_parse_conversation_spec_single():
    raw = {
        "conversation_spec_version": 1,
        "mode": "single",
        "template": "custom",
        "system_prompt": "You are helpful. {lead_name}",
    }
    spec = parse_conversation_spec(raw)
    assert isinstance(spec, ConversationSpecV1)
    assert spec.mode == "single"
    assert graph_execution_kind({"conversation_spec": raw}) == "single"


def test_parse_graph_spec_and_execution_kind():
    raw = {
        "conversation_spec_version": 1,
        "mode": "graph",
        "template": "custom",
        "entry_node_id": "alpha",
        "nodes": [
            {"id": "alpha", "label": "A", "system_prompt": "ALPHA_PROMPT body"},
            {"id": "beta", "label": "B", "system_prompt": "BETA_PROMPT body"},
        ],
        "edges": [{"from": "alpha", "to": "beta"}],
    }
    spec = parse_conversation_spec(raw)
    assert spec.mode == "graph"
    assert graph_execution_kind({"conversation_spec": raw}) == "graph"


class _GraphStubBrain(StubConversationBrain):
    async def respond(
        self,
        *,
        system_prompt: str,
        transcript: list[dict[str, str]],
        max_tokens: int | None = None,
        trace: dict | None = None,
    ) -> str:
        if "ALPHA_PROMPT" in system_prompt:
            return "alpha reply"
        if "BETA_PROMPT" in system_prompt:
            return "beta reply"
        return await super().respond(
            system_prompt=system_prompt,
            transcript=transcript,
            max_tokens=max_tokens,
            trace=trace,
        )


@pytest.mark.asyncio
async def test_custom_graph_two_nodes_linear():
    spec = {
        "conversation_spec_version": 1,
        "mode": "graph",
        "template": "custom",
        "entry_node_id": "alpha",
        "nodes": [
            {"id": "alpha", "label": "A", "system_prompt": "ALPHA_PROMPT outbound"},
            {"id": "beta", "label": "B", "system_prompt": "BETA_PROMPT continue"},
        ],
        "edges": [{"from": "alpha", "to": "beta"}],
    }
    bot_cfg = _stub_bot_config(
        conversation_spec=spec,
        initial_greeting="Opening line from entry node.",
    )
    service, _, _, _ = _build_conversation_service(brain=_GraphStubBrain(), bot_config=bot_cfg)
    conversation_id = await service.start_session("lead-001", bot_config=bot_cfg)

    state = await service.handle_turn(conversation_id, "")
    assert state["current_node"] == "alpha"
    assert state["next_node"] == "beta"
    assert "Opening line" in state["last_agent_response"]

    state = await service.handle_turn(conversation_id, "Hello from human.")
    assert state["current_node"] == "beta"
    assert state["next_node"] == "complete"
    assert state["last_agent_response"] == "beta reply"


@pytest.mark.asyncio
async def test_single_agent_mode_reuses_internal_node():
    spec = {
        "conversation_spec_version": 1,
        "mode": "single",
        "template": "custom",
        "system_prompt": "You are qualifying the prospect for a demo. {lead_name}",
    }
    bot_cfg = _stub_bot_config(conversation_spec=spec)
    service, _, _, _ = _build_conversation_service(bot_config=bot_cfg)
    conversation_id = await service.start_session("lead-001", bot_config=bot_cfg)

    state = await service.handle_turn(conversation_id, "")
    assert state["current_node"] == "__single__"
    assert state["next_node"] == "__single__"

    state = await service.handle_turn(conversation_id, "Yes, I oversee sales operations.")
    assert state["current_node"] == "__single__"
    assert state["next_node"] == "__single__"
