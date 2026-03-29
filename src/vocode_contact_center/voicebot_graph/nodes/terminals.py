from __future__ import annotations

from vocode_contact_center.settings import ContactCenterSettings
from vocode_contact_center.voicebot_graph.state import VoicebotGraphState


def set_menu(
    state: VoicebotGraphState,
    *,
    menu_name: str,
    menu_options: list[str],
    response_text: str,
) -> VoicebotGraphState:
    return {
        "active_menu": menu_name,
        "menu_options": menu_options,
        "response_text": response_text,
        "pending_prompt": response_text,
        "final_outcome": None,
        "route_decision": "complete",
    }


def complete_path(
    state: VoicebotGraphState,
    *,
    response_text: str,
    final_outcome: str,
    artifacts: dict[str, str] | None = None,
) -> VoicebotGraphState:
    return {
        "response_text": response_text,
        "pending_prompt": response_text,
        "final_outcome": final_outcome,
        "route_decision": "complete",
        "last_completed_path": state.get("current_path"),
        "active_menu": None,
        "menu_options": [],
        "pending_auth_field": None,
        "terminal_group": None,
        "artifacts": artifacts or {},
        "current_path": None,
        "root_intent": None,
        "interaction_context": None,
        "auth_status": None,
        "collected_data": {},
        "announcements_played": False,
        "genesys_requested": False,
    }


def information_store_response(settings: ContactCenterSettings) -> tuple[str, dict[str, str]]:
    return (
        f"Sure, the easiest place to check store information is on our website: {settings.information_store_website_url}.",
        {"website_url": settings.information_store_website_url},
    )


def information_products_response(settings: ContactCenterSettings) -> tuple[str, dict[str, str]]:
    return (
        f"Of course. You can review the product details in this PDF: {settings.information_products_pdf_url}.",
        {"pdf_reference": settings.information_products_pdf_url},
    )


def terminal_response_text(menu_name: str, choice: str) -> str:
    messages: dict[str, dict[str, str]] = {
        "registration_terminal": {
            "perform_registration": "Great, I'll continue with the registration for you now.",
            "registration_sms_confirmation": "No problem, I'll send the registration confirmation by SMS.",
            "generic_sms": "Sure, I'll send a general SMS with the next steps.",
        },
        "login_terminal": {
            "perform_login": "Great, I'll continue with the login process now.",
            "update_balance": "Sure, let's move ahead with the balance update.",
            "details": "Of course, I'll go over the account details next.",
        },
        "fail_terminal": {
            "communication": "Of course, I'll move you to the general communication options.",
            "generic_sms": "Sure, I'll send an SMS with the follow-up details.",
            "details": "I'll share the details I can still provide without full authentication.",
        },
        "announcements_terminal": {
            "human_agent": "Okay, I'll connect you to a human agent.",
            "call_back": "No problem, I'll arrange a call back for you.",
        },
        "feedback_terminal": {
            "human_agent": "Of course, I'll connect you to a human agent.",
            "contact": "Sure, I'll submit a contact request for follow-up.",
        },
    }
    return messages.get(menu_name, {}).get(choice, "Of course, I'll continue with that option.")
