from __future__ import annotations

from collections.abc import Callable

from loguru import logger

from vocode_contact_center.settings import ContactCenterSettings
from vocode_contact_center.sms import build_registration_confirmation_message
from vocode_contact_center.voicebot_graph.adapters.base import (
    AuthenticationAdapter,
    AuthenticationRequest,
    SmsRequest,
    SmsSender,
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
SAFE_MEMORY_FIELDS = {"full_name", "phone_number"}

_FULL_NAME_BLOCKLIST = frozenset(
    {
        "hello",
        "hi",
        "hey",
        "yes",
        "no",
        "yeah",
        "yep",
        "nope",
        "ok",
        "okay",
        "sure",
        "what",
        "help",
        "please",
        "thanks",
        "thank",
        "sorry",
        "pardon",
        "repeat",
        "uh",
        "um",
        "hmm",
        "huh",
        "still",
        "here",
        "listening",
    }
)

_FULL_NAME_REPROMPT = (
    "I need your real first and last name for registration. "
    "Please say both names clearly, for example, Jane Smith."
)


def _name_tokens(text: str) -> list[str]:
    """Letter-only tokens (hyphenated parts count as one token each)."""
    normalized = normalize_text(text)
    tokens: list[str] = []
    for raw in normalized.split():
        for part in raw.split("-"):
            letters_only = "".join(c for c in part if c.isalpha())
            if len(letters_only) >= 2:
                tokens.append(letters_only)
    return tokens


def is_plausible_full_name(user_text: str) -> bool:
    """Reject greetings, backchannels, and other non-name utterances."""
    tokens = _name_tokens(user_text)
    if len(tokens) < 2:
        return False
    if tokens[0] in _FULL_NAME_BLOCKLIST:
        return False
    if all(t in _FULL_NAME_BLOCKLIST for t in tokens):
        return False
    return True


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
    logger.info(
        "Graph authenticate session={} interaction_context={} pending_auth_field={} auth_attempts={} collected_data_keys={} latest_input={!r}",
        state.get("session_id"),
        state.get("interaction_context"),
        state.get("pending_auth_field"),
        state.get("auth_attempts", 0),
        sorted(state.get("collected_data", {}).keys()),
        state.get("latest_user_input", ""),
    )
    request = AuthenticationRequest(
        session_id=state["session_id"],
        call_context=state.get("call_context", ""),
        interaction_context=state.get("interaction_context"),
        latest_user_input=state.get("latest_user_input", ""),
        collected_data=dict(state.get("collected_data", {})),
        auth_attempts=state.get("auth_attempts", 0),
    )
    result = await auth_adapter.authenticate(request)
    logger.info(
        "Graph authenticate result session={} status={} requested_field={} normalized_data_keys={} metadata={}",
        state.get("session_id"),
        result.status,
        result.requested_field,
        sorted(result.normalized_data.keys()),
        result.metadata,
    )
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
        updates["conversation_memory"] = _merge_safe_memory(
            state.get("conversation_memory", {}),
            result.normalized_data,
        )

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
        logger.warning(
            "Graph collect_customer_input session={} called without pending_auth_field; re-entering authenticate",
            state.get("session_id"),
        )
        return {"route_decision": "interaction_authenticate"}

    normalized_value = normalize_text(state.get("latest_user_input", ""))
    logger.info(
        "Graph collect_customer_input session={} pending_field={} raw_input={!r} normalized_input={!r}",
        state.get("session_id"),
        pending_field,
        state.get("latest_user_input", ""),
        normalized_value,
    )
    if not normalized_value:
        logger.info(
            "Graph collect_customer_input session={} pending_field={} received empty normalized input; staying on current prompt",
            state.get("session_id"),
            pending_field,
        )
        return {
            "response_text": "I still need that detail before I can continue. Take your time.",
            "pending_prompt": "I still need that detail before I can continue. Take your time.",
            "route_decision": "complete",
        }

    if pending_field == "full_name" and not is_plausible_full_name(state.get("latest_user_input", "")):
        logger.info(
            "Graph collect_customer_input session={} rejected implausible full_name input={!r}",
            state.get("session_id"),
            state.get("latest_user_input", ""),
        )
        return {
            "response_text": _FULL_NAME_REPROMPT,
            "pending_prompt": _FULL_NAME_REPROMPT,
            "pending_auth_field": pending_field,
            "route_decision": "complete",
        }

    updated_data = dict(state.get("collected_data", {}))
    updated_data[pending_field] = normalized_value
    updates: VoicebotGraphState = {
        "collected_data": updated_data,
        "pending_auth_field": None,
        "route_decision": "interaction_authenticate",
    }
    logger.info(
        "Graph collect_customer_input session={} stored pending_field={} collected_data_keys={} will_reenter_authenticate=True",
        state.get("session_id"),
        pending_field,
        sorted(updated_data.keys()),
    )
    if pending_field in SAFE_MEMORY_FIELDS:
        updates["conversation_memory"] = _merge_safe_memory(
            state.get("conversation_memory", {}),
            {pending_field: normalized_value},
        )
        logger.info(
            "Graph collect_customer_input session={} updated conversation_memory keys={}",
            state.get("session_id"),
            sorted(updates["conversation_memory"].keys()),
        )
    return updates


async def sms_confirmation(
    state: VoicebotGraphState,
    sms_sender: SmsSender,
    settings: ContactCenterSettings,
    *,
    defer_sms: bool = False,
    schedule_background_sms: Callable[[SmsRequest], None] | None = None,
) -> VoicebotGraphState:
    updated_data = dict(state.get("collected_data", {}))
    phone_number = updated_data.get("phone_number", "").strip()
    adapter_results = dict(state.get("adapter_results", {}))

    if not phone_number:
        adapter_results["sms"] = {
            "status": "failed",
            "metadata": {
                "provider": "application",
                "error_message": "No phone number was available for the confirmation SMS.",
            },
        }
        return {
            "auth_status": "sms_failed",
            "response_prefix": (
                "I couldn't send the SMS confirmation because I don't have a phone number on file yet. "
            ),
            "artifacts": {"sms_status": "failed"},
            "adapter_results": adapter_results,
            "terminal_group": (
                "registration_terminal"
                if state.get("interaction_context") == "registration"
                else "login_terminal"
            ),
            "route_decision": "interaction_terminal",
        }

    sms_request = SmsRequest(
        session_id=state["session_id"],
        recipient_phone_number=phone_number,
        message=build_registration_confirmation_message(settings, updated_data),
        context="registration_confirmation",
        metadata=dict(state.get("metadata", {})),
    )

    if defer_sms and schedule_background_sms is not None:
        schedule_background_sms(sms_request)
        adapter_results["sms"] = {
            "status": "pending",
            "metadata": {
                "provider": "deferred",
                "note": "SMS send runs in the background after the verbal acknowledgment.",
            },
        }
        return {
            "collected_data": updated_data,
            "auth_status": "success",
            "response_prefix": "I'm sending a confirmation text to your phone now. ",
            "artifacts": {"sms_status": "pending"},
            "adapter_results": adapter_results,
            "terminal_group": (
                "registration_terminal"
                if state.get("interaction_context") == "registration"
                else "login_terminal"
            ),
            "route_decision": "interaction_terminal",
        }

    sms_result = await sms_sender.send(sms_request)
    adapter_results["sms"] = {
        "status": sms_result.status,
        "metadata": {
            **sms_result.metadata,
            **(
                {"provider_message_id": sms_result.provider_message_id}
                if sms_result.provider_message_id
                else {}
            ),
            **(
                {"error_message": sms_result.error_message}
                if sms_result.error_message
                else {}
            ),
        },
    }

    if sms_result.status == "sent":
        updated_data["sms_confirmed"] = "true"
        artifacts = {"sms_status": "sent"}
        if sms_result.provider_message_id:
            artifacts["sms_message_id"] = sms_result.provider_message_id
        response_prefix = "I've sent the SMS confirmation step through, so we can keep moving. "
        auth_status = "success"
    else:
        artifacts = {"sms_status": "failed"}
        response_prefix = (
            "I couldn't send the SMS confirmation just yet, but I can still help with the next step here. "
        )
        auth_status = "sms_failed"

    return {
        "collected_data": updated_data,
        "auth_status": auth_status,
        "response_prefix": response_prefix,
        "artifacts": artifacts,
        "adapter_results": adapter_results,
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


def _merge_safe_memory(
    existing_memory: dict[str, str],
    values: dict[str, str],
) -> dict[str, str]:
    updated_memory = dict(existing_memory)
    for field_name, field_value in values.items():
        if field_name in SAFE_MEMORY_FIELDS and field_value:
            updated_memory[field_name] = field_value
    return updated_memory
