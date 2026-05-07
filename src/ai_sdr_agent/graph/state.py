from __future__ import annotations

from typing import Any, Literal, TypedDict

try:
    from typing import NotRequired
except ImportError:
    from typing_extensions import NotRequired

from ai_sdr_agent.graph.spec import SINGLE_AGENT_NODE_ID, graph_execution_kind


CallOutcome = Literal[
    "meeting_booked",
    "follow_up_needed",
    "not_interested",
    "no_answer",
    "voicemail",
]


class TranscriptMessage(TypedDict):
    role: Literal["human", "agent"]
    content: str


class SlotPayload(TypedDict):
    slot_id: str
    start_time: str
    end_time: str
    label: str


class BotConfigDict(TypedDict, total=False):
    bot_id: str
    llm_provider: str
    llm_model_name: str
    llm_temperature: float
    llm_max_tokens: int
    openai_api_key: str | None
    anthropic_api_key: str | None
    groq_api_key: str | None
    elevenlabs_api_key: str | None
    elevenlabs_voice_id: str | None
    elevenlabs_model_id: str
    deepgram_api_key: str | None
    deepgram_model: str
    deepgram_language: str
    twilio_account_sid: str | None
    twilio_auth_token: str | None
    twilio_phone_number: str | None
    max_call_turns: int
    max_objection_attempts: int
    max_qualify_attempts: int
    max_booking_attempts: int
    sales_rep_name: str
    prompt_greeting: str | None
    prompt_qualify: str | None
    prompt_pitch: str | None
    prompt_objection: str | None
    prompt_booking: str | None
    prompt_wrapup: str | None
    conversation_spec: dict[str, Any] | None


class ConversationState(TypedDict):
    lead_id: str
    lead_name: str
    lead_email: str
    phone_number: str
    company: str
    calendar_id: str
    lead_context: str
    bot_config: BotConfigDict
    transcript: list[TranscriptMessage]
    current_node: str
    next_node: str
    route_decision: str
    turn_count: int
    last_human_message: str
    last_agent_response: str
    is_decision_maker: bool | None
    budget_confirmed: bool | None
    timeline: str | None
    pain_points: list[str]
    available_slots: list[SlotPayload]
    proposed_slot: str | None
    meeting_booked: bool
    meeting_link: str | None
    objection_count: int
    qualify_attempts: int
    booking_attempts: int
    call_outcome: CallOutcome
    follow_up_action: str | None
    qualification_notes: dict[str, str | bool | None]
    metadata: dict[str, str]
    # Custom graph mode: consecutive self-loop completions per node id (for loop_min/max_turns).
    graph_node_streaks: NotRequired[dict[str, int]]
    # Custom graph / single mode: next utterance index per node id for reply_turn_modes scheduling.
    graph_node_utterance_index: NotRequired[dict[str, int]]


_DEFAULT_BOT_CONFIG: BotConfigDict = {
    "bot_id": "",
    "llm_provider": "groq",
    "llm_model_name": "llama-3.3-70b-versatile",
    "llm_temperature": 0.4,
    "llm_max_tokens": 220,
    "openai_api_key": None,
    "anthropic_api_key": None,
    "groq_api_key": None,
    "elevenlabs_api_key": None,
    "elevenlabs_voice_id": None,
    "elevenlabs_model_id": "eleven_turbo_v2",
    "deepgram_api_key": None,
    "deepgram_model": "nova-2",
    "deepgram_language": "en-US",
    "twilio_account_sid": None,
    "twilio_auth_token": None,
    "twilio_phone_number": None,
    "max_call_turns": 12,
    "max_objection_attempts": 2,
    "max_qualify_attempts": 3,
    "max_booking_attempts": 3,
    "sales_rep_name": "Sales Team",
    "prompt_greeting": None,
    "prompt_qualify": None,
    "prompt_pitch": None,
    "prompt_objection": None,
    "prompt_booking": None,
    "prompt_wrapup": None,
    "conversation_spec": None,
}


def _initial_route_target(bot_config: BotConfigDict | None) -> str:
    cfg: BotConfigDict = bot_config or dict(_DEFAULT_BOT_CONFIG)
    kind = graph_execution_kind(dict(cfg))
    if kind == "sdr":
        return "greeting"
    if kind == "single":
        return SINGLE_AGENT_NODE_ID
    spec = cfg.get("conversation_spec") or {}
    entry = spec.get("entry_node_id")
    if isinstance(entry, str) and entry:
        return entry
    return "greeting"


def build_initial_state(
    *,
    lead_id: str,
    lead_name: str,
    lead_email: str,
    phone_number: str,
    company: str,
    calendar_id: str,
    lead_context: str,
    available_slots: list[SlotPayload],
    bot_config: BotConfigDict | None = None,
) -> ConversationState:
    merged = dict(_DEFAULT_BOT_CONFIG)
    if bot_config:
        merged.update(bot_config)
    route = _initial_route_target(merged)
    return {
        "lead_id": lead_id,
        "lead_name": lead_name,
        "lead_email": lead_email,
        "phone_number": phone_number,
        "company": company,
        "calendar_id": calendar_id,
        "lead_context": lead_context,
        "bot_config": merged,
        "transcript": [],
        "current_node": "start",
        "next_node": route,
        "route_decision": route,
        "turn_count": 0,
        "last_human_message": "",
        "last_agent_response": "",
        "is_decision_maker": None,
        "budget_confirmed": None,
        "timeline": None,
        "pain_points": [],
        "available_slots": available_slots,
        "proposed_slot": None,
        "meeting_booked": False,
        "meeting_link": None,
        "objection_count": 0,
        "qualify_attempts": 0,
        "booking_attempts": 0,
        "call_outcome": "follow_up_needed",
        "follow_up_action": None,
        "qualification_notes": {},
        "metadata": {},
        "graph_node_streaks": {},
        "graph_node_utterance_index": {},
    }
