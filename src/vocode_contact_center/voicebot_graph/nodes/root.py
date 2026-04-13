from __future__ import annotations

from loguru import logger

from vocode_contact_center.voicebot_graph.intents import (
    classify_global_navigation,
    classify_root_intent,
)
from vocode_contact_center.voicebot_graph.state import VoicebotGraphState, reset_to_root_menu


ROOT_MENU_PROMPT = (
    "I can help with PeopleCert information from our website or documents, your account registration or sign-in, "
    "announcements, or feedback and contact options. Which would you like?"
)
ROOT_MENU_OPTIONS = ["information", "interaction", "announcements", "feedback"]


def route_turn(state: VoicebotGraphState) -> VoicebotGraphState:
    latest_user_input = state.get("latest_user_input", "")
    navigation = classify_global_navigation(latest_user_input)
    root_intent = classify_root_intent(latest_user_input)
    current_path = state.get("current_path")
    pending_auth_field = state.get("pending_auth_field")
    active_menu = state.get("active_menu")

    logger.info(
        "Graph route_turn session={} input={!r} current_path={} active_menu={} pending_auth_field={} navigation={} root_intent={}",
        state.get("session_id"),
        latest_user_input,
        current_path,
        active_menu,
        pending_auth_field,
        navigation,
        root_intent,
    )

    if navigation == "main_menu":
        logger.info(
            "Graph route_turn redirecting to main menu from explicit navigation session={} input={!r}",
            state.get("session_id"),
            latest_user_input,
        )
        return {"route_decision": "global_main_menu"}

    # Collect full name before keyword cross-flow detection so substrings like "other"
    # (information intent) do not cancel registration mid-prompt.
    if pending_auth_field == "full_name":
        logger.info(
            "Graph route_turn continuing auth field collection session={} pending_auth_field={} input={!r}",
            state.get("session_id"),
            pending_auth_field,
            latest_user_input,
        )
        return {"route_decision": "interaction_customer_input"}

    if _should_redirect_to_main_menu(state, requested_root_intent=root_intent):
        logger.info(
            "Graph route_turn redirecting to main menu from cross-flow request session={} current_path={} requested_root_intent={} input={!r}",
            state.get("session_id"),
            current_path,
            root_intent,
            latest_user_input,
        )
        return {"route_decision": "global_main_menu"}

    if pending_auth_field:
        logger.info(
            "Graph route_turn continuing auth field collection session={} pending_auth_field={} input={!r}",
            state.get("session_id"),
            pending_auth_field,
            latest_user_input,
        )
        return {"route_decision": "interaction_customer_input"}

    if active_menu == "root_intent":
        return {"route_decision": "root_intent"}
    if active_menu == "interaction_entry":
        return {"route_decision": "interaction_entry"}
    if active_menu in {"info_selection", "change_information", "information_products"}:
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
    logger.info(
        "Graph resolve_root_intent session={} input={!r} classified_root_intent={}",
        state.get("session_id"),
        state.get("latest_user_input", ""),
        root_intent,
    )
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
    logger.info(
        "Graph return_to_main_menu session={} previous_path={} previous_menu={} pending_auth_field={} conversation_memory_keys={}",
        state.get("session_id"),
        state.get("current_path"),
        state.get("active_menu"),
        state.get("pending_auth_field"),
        sorted(state.get("conversation_memory", {}).keys()),
    )
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
