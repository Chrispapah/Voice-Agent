import asyncio

from fastapi import FastAPI
from fastapi.testclient import TestClient

from vocode_contact_center.realtime_worker import (
    RealtimeSessionCreateRequest,
    RealtimeSessionManager,
    RealtimeVoiceSession,
    create_realtime_router,
)
from vocode_contact_center.settings import ContactCenterSettings


class FakeLLM:
    def __init__(self, tokens: list[str], *, delay_seconds: float = 0.0):
        self.tokens = tokens
        self.delay_seconds = delay_seconds

    async def stream_response(self, **kwargs):
        for token in self.tokens:
            if self.delay_seconds:
                await asyncio.sleep(self.delay_seconds)
            yield token


class FakeTTS:
    def __init__(self, *, delay_seconds: float = 0.0):
        self.delay_seconds = delay_seconds
        self.cancel_calls = 0

    @property
    def audio_encoding(self) -> str:
        return "linear16"

    @property
    def sample_rate(self) -> int:
        return 16000

    async def stream_tokens(self, tokens):
        async for token in tokens:
            if self.delay_seconds:
                await asyncio.sleep(self.delay_seconds)
            yield token.encode("utf-8")

    async def cancel_current_utterance(self) -> None:
        self.cancel_calls += 1


def make_settings(**overrides) -> ContactCenterSettings:
    base = {
        "elevenlabs_api_key": "eleven",
        "elevenlabs_voice_id": "voice",
        "langchain_provider": "openai",
        "openai_api_key": "openai",
        "realtime_partial_response_min_words": 1,
        "realtime_partial_response_min_chars": 1,
    }
    base.update(overrides)
    return ContactCenterSettings(**base)


def test_realtime_manager_creates_websocket_sessions_for_new_channel():
    manager = RealtimeSessionManager(
        make_settings(),
        llm=FakeLLM(["hello"]),
        tts_factory=FakeTTS,
        legacy_telephony_available=True,
    )

    response = manager.create_session(RealtimeSessionCreateRequest(call_context="Customer is online"))

    assert response.transport == "websocket"
    assert response.websocket_path.endswith(f"/{response.session_id}/ws")
    assert response.output_audio_encoding == "linear16"
    assert response.legacy_telephony_available is True


def test_realtime_router_rejects_session_creation_when_not_ready():
    app = FastAPI()
    manager = RealtimeSessionManager(
        make_settings(),
        llm=FakeLLM(["hello"]),
        tts_factory=FakeTTS,
    )
    app.include_router(
        create_realtime_router(
            make_settings(),
            manager=manager,
            realtime_ready=False,
        )
    )

    client = TestClient(app)
    response = client.post("/realtime/sessions", json={})

    assert response.status_code == 503
    assert "missing_runtime_values" in response.json()["detail"]


def test_realtime_voice_session_commits_only_final_turns():
    captured_events = []
    settings = make_settings()
    session = RealtimeVoiceSession(
        session_id="session-1",
        call_context="Realtime session metadata is unavailable.",
        metadata={},
        send_event=captured_events.append,
        llm=FakeLLM(["Hello ", "there."]),
        tts=FakeTTS(),
        settings=settings,
        metrics=RealtimeSessionManager(
            settings,
            llm=FakeLLM(["unused"]),
            tts_factory=FakeTTS,
        ).metrics,
    )

    async def run_test():
        async def send(payload):
            captured_events.append(payload)

        session.send_event = send
        await session.handle_client_event({"type": "user_partial", "text": "hello"})
        await session._assistant_task
        assert session.committed_messages == []

        await session.handle_client_event({"type": "user_final", "text": "hello"})
        await session._assistant_task

    asyncio.run(run_test())

    assert session.committed_messages[0] == ("human", "hello")
    assert session.committed_messages[1][0] == "ai"
    assert any(event["type"] == "assistant_turn_end" for event in captured_events)


def test_realtime_voice_session_interrupts_and_restarts_on_barge_in():
    captured_events = []
    settings = make_settings()
    tts = FakeTTS(delay_seconds=0.01)
    session = RealtimeVoiceSession(
        session_id="session-2",
        call_context="Realtime session metadata is unavailable.",
        metadata={},
        send_event=lambda payload: asyncio.sleep(0, result=captured_events.append(payload)),
        llm=FakeLLM(["Hello ", "world"], delay_seconds=0.02),
        tts=tts,
        settings=settings,
        metrics=RealtimeSessionManager(
            settings,
            llm=FakeLLM(["unused"]),
            tts_factory=FakeTTS,
        ).metrics,
    )

    async def run_test():
        async def send(payload):
            captured_events.append(payload)

        session.send_event = send
        await session.handle_client_event({"type": "user_final", "text": "hello"})
        await asyncio.sleep(0.03)
        await session.handle_client_event({"type": "user_partial", "text": "wait"})
        await asyncio.sleep(0.03)
        if session._assistant_task is not None:
            await session._assistant_task

    asyncio.run(run_test())

    assert any(event["type"] == "assistant_interrupted" for event in captured_events)
    assert tts.cancel_calls >= 1
    assert sum(event["type"] == "assistant_response_started" for event in captured_events) >= 2


def test_realtime_manager_default_voicebot_llm_uses_state_graph():
    captured_events = []
    settings = make_settings()
    manager = RealtimeSessionManager(
        settings,
        tts_factory=FakeTTS,
    )
    session_response = manager.create_session(RealtimeSessionCreateRequest(call_context="Realtime"))
    session = manager._sessions[session_response.session_id]

    async def run_test():
        async def send(payload):
            captured_events.append(payload)

        session.send_event = send
        await session.handle_client_event({"type": "user_final", "text": "I need information"})
        await session._assistant_task

    asyncio.run(run_test())

    turn_end_events = [event for event in captured_events if event["type"] == "assistant_turn_end"]
    assert turn_end_events
    assert "store, products, or other" in turn_end_events[-1]["text"].lower()

