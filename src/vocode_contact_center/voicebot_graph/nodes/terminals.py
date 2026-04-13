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
    primary = settings.information_store_website_url
    artifacts: dict[str, str] = {
        "website_url": primary,
        "help_faq_url": settings.peoplecert_help_faq_url,
        "olp_guidelines_windows_pdf": settings.peoplecert_olp_guidelines_pdf_windows_url,
        "olp_guidelines_mac_pdf": settings.peoplecert_olp_guidelines_pdf_mac_url,
        "take2_url": settings.peoplecert_take2_url,
        "certificate_verification_url": settings.peoplecert_certificate_verification_url,
        "corporate_membership_url": settings.peoplecert_corporate_membership_url,
        "itil4_foundation_url": settings.peoplecert_itil4_foundation_url,
    }
    return (
        (
            f"For Online Proctored Exam help, start at {primary}. "
            f"For FAQs, use {settings.peoplecert_help_faq_url}. "
            f"Web proctored candidate guidelines: Windows PDF {settings.peoplecert_olp_guidelines_pdf_windows_url}, "
            f"Mac PDF {settings.peoplecert_olp_guidelines_pdf_mac_url}. "
            f"Take2: {settings.peoplecert_take2_url}. "
            f"Certificate verification: {settings.peoplecert_certificate_verification_url}. "
            f"Corporate membership: {settings.peoplecert_corporate_membership_url}. "
            f"ITIL 4 Foundation: {settings.peoplecert_itil4_foundation_url}."
        ),
        artifacts,
    )


def information_products_response(settings: ContactCenterSettings) -> tuple[str, dict[str, str]]:
    url = (settings.information_products_pdf_url or "").strip()
    if not url:
        url = settings.peoplecert_olp_guidelines_pdf_windows_url
    return (
        f"You can review the official document here: {url}. For Mac-specific steps, see {settings.peoplecert_olp_guidelines_pdf_mac_url}.",
        {"pdf_reference": url, "olp_guidelines_mac_pdf": settings.peoplecert_olp_guidelines_pdf_mac_url},
    )


def terminal_response_text(menu_name: str, choice: str) -> str:
    messages: dict[str, dict[str, str]] = {
        "registration_terminal": {
            "perform_registration": "Great, I'll continue with the registration for you now.",
            "registration_sms_confirmation": "No problem, I'll send the registration confirmation by SMS.",
            "generic_sms": "Sure, I'll send a general SMS with the next steps.",
        },
        "login_terminal": {
            "perform_login": "Great, I'll continue with sign-in now.",
            "exam_booking_help": "I'll walk you through exams and booking on peoplecert.org next.",
            "certificates_access": "I'll guide you to certificates and results in your account next.",
            "profile_password": "I'll help with profile and password reset steps next.",
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
