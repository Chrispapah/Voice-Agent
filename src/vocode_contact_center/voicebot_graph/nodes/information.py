from __future__ import annotations

from vocode_contact_center.settings import ContactCenterSettings
from vocode_contact_center.voicebot_graph.intents import (
    classify_change_information_choice,
    classify_information_choice,
)
from vocode_contact_center.voicebot_graph.nodes.terminals import (
    complete_path,
    information_products_response,
    information_store_response,
    set_menu,
)
from vocode_contact_center.voicebot_graph.state import VoicebotGraphState


INFO_MENU_PROMPT = (
    "For information, say Store, Products, or Other."
)
CHANGE_INFORMATION_PROMPT = (
    "You chose Other. If you want to change the information request, say Change Information."
)


def handle_information(state: VoicebotGraphState, settings: ContactCenterSettings) -> VoicebotGraphState:
    active_menu = state.get("active_menu")
    if active_menu == "change_information":
        return _handle_change_information(state)

    if active_menu != "info_selection":
        return set_menu(
            state,
            menu_name="info_selection",
            menu_options=["store", "products", "other"],
            response_text=INFO_MENU_PROMPT,
        )

    choice = classify_information_choice(state.get("latest_user_input", ""))
    if choice == "store":
        response_text, artifacts = information_store_response(settings)
        return complete_path(
            state,
            response_text=response_text,
            final_outcome="website",
            artifacts=artifacts,
        )
    if choice == "products":
        response_text, artifacts = information_products_response(settings)
        return complete_path(
            state,
            response_text=response_text,
            final_outcome="pdf",
            artifacts=artifacts,
        )
    if choice == "other":
        return set_menu(
            state,
            menu_name="change_information",
            menu_options=["change_information", "cancel"],
            response_text=CHANGE_INFORMATION_PROMPT,
        )

    return set_menu(
        state,
        menu_name="info_selection",
        menu_options=["store", "products", "other"],
        response_text=INFO_MENU_PROMPT,
    )


def _handle_change_information(state: VoicebotGraphState) -> VoicebotGraphState:
    choice = classify_change_information_choice(state.get("latest_user_input", ""))
    if choice == "change_information":
        return set_menu(
            state,
            menu_name="info_selection",
            menu_options=["store", "products", "other"],
            response_text="Okay, let's try again. Say Store, Products, or Other.",
        )
    if choice == "cancel":
        return complete_path(
            state,
            response_text="Okay, we can leave the information path here.",
            final_outcome="information_cancelled",
        )
    return set_menu(
        state,
        menu_name="change_information",
        menu_options=["change_information", "cancel"],
        response_text=CHANGE_INFORMATION_PROMPT,
    )
