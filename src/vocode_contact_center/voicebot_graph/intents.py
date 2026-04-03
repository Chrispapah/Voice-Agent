from __future__ import annotations

import unicodedata
from typing import Iterable


ROOT_INTENTS = ("information", "interaction", "announcements", "feedback")
INTERACTION_CONTEXTS = ("registration", "login")


def normalize_text(text: str) -> str:
    lowered = text.strip().lower()
    normalized = unicodedata.normalize("NFD", lowered)
    without_marks = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return " ".join(without_marks.split())


def contains_any(text: str, keywords: Iterable[str]) -> bool:
    normalized = normalize_text(text)
    return any(keyword in normalized for keyword in keywords)


def classify_root_intent(text: str) -> str | None:
    normalized = normalize_text(text)

    if contains_any(
        normalized,
        (
            "ανακοινωσ",
            "announcement",
            "announcements",
            "news",
            "updates",
        ),
    ):
        return "announcements"

    if contains_any(
        normalized,
        (
            "εγγραφ",
            "συνδεσ",
            "κωδικ",
            "λογαριασ",
            "συνομιλια",
            "interaction",
            "register",
            "registration",
            "login",
            "sign in",
            "sign-in",
            "password",
            "account",
            "authenticate",
            "authentication",
        ),
    ):
        return "interaction"

    if contains_any(
        normalized,
        (
            "πληροφορι",
            "καταστημα",
            "προϊον",
            "προιον",
            "άλλο",
            "αλλο",
            "information",
            "info",
            "store",
            "product",
            "products",
            "other",
            "website",
            "pdf",
        ),
    ):
        return "information"

    if contains_any(
        normalized,
        (
            "πιστευ",
            "feedback",
            "survey",
            "contact options",
            "back to chat",
            "contact me",
        ),
    ):
        return "feedback"

    return None


def classify_global_navigation(text: str) -> str | None:
    normalized = normalize_text(text)
    if contains_any(
        normalized,
        (
            "cancel",
            "cancel verification",
            "cancel registration",
            "never mind",
            "forget it",
            "go back",
            "main menu",
            "menu",
            "start over",
            "abort",
            "exit",
            "home",
            "ακυρο",
            "σταματα",
            "σταματησε",
            "πισω",
            "κεντρικο μενου",
        ),
    ):
        return "main_menu"
    return None


def classify_interaction_context(text: str) -> str | None:
    normalized = normalize_text(text)
    if contains_any(
        normalized,
        (
            "εγγραφ",
            "register",
            "registration",
            "sign up",
            "sign-up",
            "create account",
            "new account",
        ),
    ):
        return "registration"
    if contains_any(
        normalized,
        (
            "συνδεσ",
            "κωδικ",
            "login",
            "log in",
            "sign in",
            "sign-in",
            "password",
            "account access",
            "other",
        ),
    ):
        return "login"
    return None


def classify_information_choice(text: str) -> str | None:
    normalized = normalize_text(text)
    if contains_any(normalized, ("καταστημα", "store", "shop", "website")):
        return "store"
    if contains_any(normalized, ("προϊον", "προιον", "product", "products", "pdf")):
        return "products"
    if contains_any(normalized, ("άλλο", "αλλο", "other")):
        return "other"
    return None


def classify_change_information_choice(text: str) -> str | None:
    normalized = normalize_text(text)
    if contains_any(
        normalized,
        (
            "αλλαγη πληροφορια",
            "change information",
            "change info",
            "yes",
            "ναι",
            "correct",
            "go back",
        ),
    ):
        return "change_information"
    if contains_any(normalized, ("no", "οχι", "cancel", "stop")):
        return "cancel"
    return None


def classify_announcements_continue_choice(text: str) -> str | None:
    normalized = normalize_text(text)
    if contains_any(normalized, ("συνεχιζει", "continue", "proceed", "yes", "ναι")):
        return "continue"
    if contains_any(normalized, ("no", "οχι", "stop", "end")):
        return "stop"
    return None


def classify_feedback_question_choice(text: str) -> str | None:
    normalized = normalize_text(text)
    if contains_any(normalized, ("yes", "ναι", "back to chat", "chat")):
        return "back_to_chat"
    if contains_any(normalized, ("no", "οχι", "support", "agent", "contact")):
        return "genesys"
    return None


def classify_terminal_choice(text: str, menu_name: str) -> str | None:
    normalized = normalize_text(text)

    menu_keywords: dict[str, dict[str, tuple[str, ...]]] = {
        "registration_terminal": {
            "perform_registration": ("registration", "perform registration", "complete registration"),
            "registration_sms_confirmation": (
                "sms confirmation",
                "registration sms",
                "confirm by sms",
            ),
            "generic_sms": ("generic sms", "send sms", "sms"),
        },
        "login_terminal": {
            "perform_login": ("login", "perform login", "sign in"),
            "update_balance": ("ενημερωση υπολοιπου", "update balance", "balance"),
            "details": ("λεπτομερειες", "details", "account details"),
        },
        "fail_terminal": {
            "communication": ("επικοινωνια", "communication", "contact options"),
            "generic_sms": ("sms", "generic sms", "text message"),
            "details": ("λεπτομερειες", "details", "more details"),
        },
        "announcements_terminal": {
            "human_agent": ("human agent", "agent", "representative"),
            "call_back": ("call back", "callback"),
        },
        "feedback_terminal": {
            "human_agent": ("human agent", "agent", "representative"),
            "contact": ("contact", "contact me", "communication"),
        },
    }

    options = menu_keywords.get(menu_name, {})
    for choice, keywords in options.items():
        if contains_any(normalized, keywords):
            return choice
    return None
