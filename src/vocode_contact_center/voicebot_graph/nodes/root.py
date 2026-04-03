from __future__ import annotations

from vocode_contact_center.voicebot_graph.intents import (
    classify_global_navigation,
    classify_root_intent,
)
from vocode_contact_center.voicebot_graph.state import VoicebotGraphState, reset_to_root_menu


ROOT_MENU_PROMPT = (
    "I can help with general information, account support, announcements, or feedback and contact options. Which one would you like to start with?"
)
ROOT_MENU_OPTIONS = ["information", "interaction", "announcements", "feedback"]


def route_turn(state: VoicebotGraphState) -> VoicebotGraphState:
    latest_user_input = state.get("latest_user_input", "")
    if classify_global_navigation(latest_user_input) == "main_menu":
        return {"route_decision": "global_main_menu"}

    root_intent = classify_root_intent(latest_user_input)
    if _should_redirect_to_main_menu(state, requested_root_intent=root_intent):
        return {"route_decision": "global_main_menu"}

    if state.get("pending_auth_field"):
        return {"route_decision": "interaction_customer_input"}

    active_menu = state.get("active_menu")
    if active_menu == "root_intent":
        return {"route_decision": "root_intent"}
    if active_menu == "interaction_entry":
        return {"route_decision": "interaction_entry"}
    if active_menu in {"info_selection", "change_information"}:
        return {"route_decision": "information"}
    if active_menu in {"registration_terminal", "login_terminal", "fail_terminal"}:
        return {"route_decision": "interaction_terminal"}
    if active_menu in {"announcements_continue", "announcements_terminal"}:
        return {"route_decision": "announcements"}
    if active_menu in {"feedback_question", "feedback_terminal"}:
        return {"route_decision": "feedback"}

    return {"route_decision": "root_intent"}


def resolve_root_intent(state: VoicebotGraphState) -> VoicebotGraphState:
    root_intent = classify_root_intent(state.get("latest_user_input", ""))
    if root_intent is None:
        return {
            "active_menu": "root_intent",
            "menu_options": ROOT_MENU_OPTIONS,
            "response_text": ROOT_MENU_PROMPT,
            "pending_prompt": ROOT_MENU_PROMPT,
            "route_decision": "complete",
        }

    updates: VoicebotGraphState = {
        "root_intent": root_intent,
        "current_path": root_intent,
        "active_menu": None,
        "menu_options": [],
    }
    if root_intent == "information":
        updates["route_decision"] = "information"
    elif root_intent == "interaction":
        updates["route_decision"] = "interaction_entry"
    elif root_intent == "announcements":
        updates["route_decision"] = "announcements"
    else:
        updates["route_decision"] = "feedback"
    return updates


def return_to_main_menu(state: VoicebotGraphState) -> VoicebotGraphState:
    return reset_to_root_menu(
        state,
        response_text=(
            "No problem. I've cancelled that process and taken you back to the main menu. "
            f"{ROOT_MENU_PROMPT}"
        ),
    )


def _should_redirect_to_main_menu(
    state: VoicebotGraphState,
    *,
    requested_root_intent: str | None,
) -> bool:
    if requested_root_intent is None:
        return False

    current_path = state.get("current_path")
    if not current_path:
        return False

    return requested_root_intent != current_path
