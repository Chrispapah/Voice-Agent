from __future__ import annotations

from vocode_contact_center.voicebot_graph.adapters.base import (
    AuthenticationAdapter,
    AuthenticationRequest,
)
from vocode_contact_center.voicebot_graph.intents import (
    classify_interaction_context,
    classify_terminal_choice,
    normalize_text,
)
from vocode_contact_center.voicebot_graph.nodes.terminals import (
    complete_path,
    set_menu,
    terminal_response_text,
)
from vocode_contact_center.voicebot_graph.state import VoicebotGraphState


INTERACTION_ENTRY_PROMPT = (
    "I can help you create a new account or access an existing one. Would you like registration or login support?"
)


def handle_interaction_entry(state: VoicebotGraphState) -> VoicebotGraphState:
    context = state.get("interaction_context") or classify_interaction_context(
        state.get("latest_user_input", "")
    )
    if context is None:
        return set_menu(
            state,
            menu_name="interaction_entry",
            menu_options=["registration", "login"],
            response_text=INTERACTION_ENTRY_PROMPT,
        )
    return {
        "interaction_context": context,
        "auth_status": "pending",
        "active_menu": None,
        "menu_options": [],
        "route_decision": "interaction_authenticate",
    }


async def authenticate(
    state: VoicebotGraphState,
    auth_adapter: AuthenticationAdapter,
) -> VoicebotGraphState:
    request = AuthenticationRequest(
        session_id=state["session_id"],
        call_context=state.get("call_context", ""),
        interaction_context=state.get("interaction_context"),
        latest_user_input=state.get("latest_user_input", ""),
        collected_data=dict(state.get("collected_data", {})),
        auth_attempts=state.get("auth_attempts", 0),
    )
    result = await auth_adapter.authenticate(request)
    updates: VoicebotGraphState = {
        "auth_status": result.status,
        "auth_attempts": state.get("auth_attempts", 0) + 1,
        "adapter_results": {
            "authentication": {
                "status": result.status,
                "metadata": result.metadata,
            }
        },
    }

    if result.normalized_data:
        updated_data = dict(state.get("collected_data", {}))
        updated_data.update(result.normalized_data)
        updates["collected_data"] = updated_data

    if result.status == "needs_customer_input":
        updates["pending_auth_field"] = result.requested_field
        updates["response_text"] = result.prompt
        updates["pending_prompt"] = result.prompt
        updates["route_decision"] = "complete"
        return updates

    if result.status == "needs_sms_confirmation":
        updates["route_decision"] = "interaction_sms_confirmation"
        return updates

    if result.status == "success":
        updates["terminal_group"] = (
            "registration_terminal"
            if state.get("interaction_context") == "registration"
            else "login_terminal"
        )
        updates["route_decision"] = "interaction_terminal"
        return updates

    updates["terminal_group"] = "fail_terminal"
    updates["route_decision"] = "interaction_terminal"
    return updates


def collect_customer_input(state: VoicebotGraphState) -> VoicebotGraphState:
    pending_field = state.get("pending_auth_field")
    if not pending_field:
        return {"route_decision": "interaction_authenticate"}

    normalized_value = normalize_text(state.get("latest_user_input", ""))
    if not normalized_value:
        return {
            "response_text": "I still need that detail before I can continue. Take your time.",
            "pending_prompt": "I still need that detail before I can continue. Take your time.",
            "route_decision": "complete",
        }

    updated_data = dict(state.get("collected_data", {}))
    updated_data[pending_field] = normalized_value
    return {
        "collected_data": updated_data,
        "pending_auth_field": None,
        "route_decision": "interaction_authenticate",
    }


def sms_confirmation(state: VoicebotGraphState) -> VoicebotGraphState:
    updated_data = dict(state.get("collected_data", {}))
    updated_data["sms_confirmed"] = "true"
    return {
        "collected_data": updated_data,
        "auth_status": "success",
        "response_prefix": "I've sent the SMS confirmation step through, so we can keep moving. ",
        "artifacts": {"sms_status": "sent"},
        "terminal_group": (
            "registration_terminal"
            if state.get("interaction_context") == "registration"
            else "login_terminal"
        ),
        "route_decision": "interaction_terminal",
    }


def handle_terminal_menu(state: VoicebotGraphState) -> VoicebotGraphState:
    menu_name = state.get("active_menu") or state.get("terminal_group") or "fail_terminal"

    if state.get("active_menu") != menu_name:
        return _prompt_for_terminal_menu(state, menu_name)

    choice = classify_terminal_choice(state.get("latest_user_input", ""), menu_name)
    if choice is None:
        return _prompt_for_terminal_menu(state, menu_name)

    response_text = terminal_response_text(menu_name, choice)
    return complete_path(
        state,
        response_text=response_text,
        final_outcome=choice,
        artifacts=state.get("artifacts"),
    )


def _prompt_for_terminal_menu(state: VoicebotGraphState, menu_name: str) -> VoicebotGraphState:
    prompts = {
        "registration_terminal": (
            "You're all set to continue. I can complete the registration, send the registration SMS confirmation, or send a general SMS with the next steps. Which would you prefer?"
        ),
        "login_terminal": (
            "You're verified. I can continue with login, help with a balance update, or go over the account details. What would you like to do next?"
        ),
        "fail_terminal": (
            "It looks like authentication didn't fully complete. I can still help with general communication options, send an SMS, or share general details. Which would you like?"
        ),
    }
    prefix = state.get("response_prefix", "")
    return set_menu(
        state,
        menu_name=menu_name,
        menu_options=list(_terminal_options(menu_name)),
        response_text=f"{prefix}{prompts[menu_name]}",
    )


def _terminal_options(menu_name: str) -> tuple[str, ...]:
    options = {
        "registration_terminal": (
            "perform_registration",
            "registration_sms_confirmation",
            "generic_sms",
        ),
        "login_terminal": ("perform_login", "update_balance", "details"),
        "fail_terminal": ("communication", "generic_sms", "details"),
    }
    return options[menu_name]
