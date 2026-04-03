from __future__ import annotations

from copy import deepcopy
from typing import Any, TypedDict


class VoicebotArtifacts(TypedDict, total=False):
    website_url: str
    pdf_reference: str
    sms_status: str
    sms_message_id: str
    announcements_status: str


class VoicebotGraphState(TypedDict, total=False):
    session_id: str
    call_context: str
    metadata: dict[str, str]
    latest_user_input: str
    current_path: str | None
    root_intent: str | None
    interaction_context: str | None
    auth_status: str | None
    auth_attempts: int
    pending_auth_field: str | None
    pending_prompt: str | None
    active_menu: str | None
    menu_options: list[str]
    terminal_group: str | None
    final_outcome: str | None
    response_text: str
    response_prefix: str
    route_decision: str
    collected_data: dict[str, str]
    conversation_memory: dict[str, str]
    adapter_results: dict[str, Any]
    artifacts: VoicebotArtifacts
    announcements_played: bool
    genesys_requested: bool
    last_user_selection: str | None
    last_completed_path: str | None


def initial_graph_state(
    *,
    session_id: str,
    call_context: str,
    metadata: dict[str, str] | None = None,
) -> VoicebotGraphState:
    return {
        "session_id": session_id,
        "call_context": call_context,
        "metadata": deepcopy(metadata or {}),
        "latest_user_input": "",
        "current_path": None,
        "root_intent": None,
        "interaction_context": None,
        "auth_status": None,
        "auth_attempts": 0,
        "pending_auth_field": None,
        "pending_prompt": None,
        "active_menu": None,
        "menu_options": [],
        "terminal_group": None,
        "final_outcome": None,
        "response_text": "",
        "response_prefix": "",
        "route_decision": "root_intent",
        "collected_data": {},
        "conversation_memory": {},
        "adapter_results": {},
        "artifacts": {},
        "announcements_played": False,
        "genesys_requested": False,
        "last_user_selection": None,
        "last_completed_path": None,
    }


def clone_state(state: VoicebotGraphState) -> VoicebotGraphState:
    return deepcopy(state)


def clear_turn_fields(state: VoicebotGraphState) -> VoicebotGraphState:
    updated = clone_state(state)
    updated["latest_user_input"] = ""
    updated["response_text"] = ""
    updated["response_prefix"] = ""
    updated["pending_prompt"] = None
    updated["adapter_results"] = {}
    updated["artifacts"] = {}
    updated["last_user_selection"] = None
    return updated


def reset_path_state(state: VoicebotGraphState) -> VoicebotGraphState:
    updated = clear_turn_fields(state)
    updated["current_path"] = None
    updated["root_intent"] = None
    updated["interaction_context"] = None
    updated["auth_status"] = None
    updated["auth_attempts"] = 0
    updated["pending_auth_field"] = None
    updated["active_menu"] = None
    updated["menu_options"] = []
    updated["terminal_group"] = None
    updated["final_outcome"] = None
    updated["collected_data"] = {}
    updated["announcements_played"] = False
    updated["genesys_requested"] = False
    return updated


def reset_to_root_menu(
    state: VoicebotGraphState,
    *,
    response_text: str,
    final_outcome: str = "cancelled_to_main_menu",
) -> VoicebotGraphState:
    updated = reset_path_state(state)
    updated["active_menu"] = "root_intent"
    updated["menu_options"] = ["information", "interaction", "announcements", "feedback"]
    updated["final_outcome"] = final_outcome
    updated["response_text"] = response_text
    updated["pending_prompt"] = response_text
    updated["route_decision"] = "complete"
    updated["adapter_results"] = {}
    updated["artifacts"] = {}
    return updated
