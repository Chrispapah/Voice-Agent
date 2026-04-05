from __future__ import annotations

from typing import Protocol

from vocode_contact_center.product_knowledge import ProductKnowledgeAnswer
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
    "I can help with the PeopleCert website, answers from our help document, or something else. Which would you like?"
)
CHANGE_INFORMATION_PROMPT = (
    "No problem. If you'd like, we can go back and choose a different type of information request."
)
PRODUCT_QUESTION_PROMPT = (
    "I can answer questions from our configured PeopleCert help document. What would you like to know? "
    "You can also say change information to pick another option."
)


class ProductInformationResponder(Protocol):
    def is_configured(self) -> bool:
        ...

    async def answer_question(self, question: str) -> ProductKnowledgeAnswer:
        ...


async def handle_information(
    state: VoicebotGraphState,
    settings: ContactCenterSettings,
    product_information: ProductInformationResponder,
) -> VoicebotGraphState:
    active_menu = state.get("active_menu")
    if active_menu == "change_information":
        return _handle_change_information(state)
    if active_menu == "information_products":
        return await _handle_product_information_question(state, product_information)

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
        if product_information.is_configured():
            return set_menu(
                state,
                menu_name="information_products",
                menu_options=["ask_product_question", "change_information"],
                response_text=PRODUCT_QUESTION_PROMPT,
            )
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
            response_text=f"{CHANGE_INFORMATION_PROMPT} Just say Change Information to start again, or say cancel if you'd rather stop here.",
        )

    return set_menu(
        state,
        menu_name="info_selection",
        menu_options=["store", "products", "other"],
        response_text=INFO_MENU_PROMPT,
    )


async def _handle_product_information_question(
    state: VoicebotGraphState,
    product_information: ProductInformationResponder,
) -> VoicebotGraphState:
    choice = classify_change_information_choice(state.get("latest_user_input", ""))
    if choice == "change_information":
        return set_menu(
            state,
            menu_name="info_selection",
            menu_options=["store", "products", "other"],
            response_text=f"Of course. {INFO_MENU_PROMPT}",
        )
    if choice == "cancel":
        return complete_path(
            state,
            response_text="That's fine. We can leave the document questions there for now.",
            final_outcome="information_cancelled",
        )

    result = await product_information.answer_question(state.get("latest_user_input", ""))
    follow_up = (
        " You can ask another question, or say change information to choose something else."
    )
    return set_menu(
        state,
        menu_name="information_products",
        menu_options=["ask_product_question", "change_information"],
        response_text=f"{result.text}{follow_up}",
    ) | {"artifacts": result.artifacts}


def _handle_change_information(state: VoicebotGraphState) -> VoicebotGraphState:
    choice = classify_change_information_choice(state.get("latest_user_input", ""))
    if choice == "change_information":
        return set_menu(
            state,
            menu_name="info_selection",
            menu_options=["store", "products", "other"],
            response_text=f"Of course, let's try again. {INFO_MENU_PROMPT}",
        )
    if choice == "cancel":
        return complete_path(
            state,
            response_text="That's absolutely fine. We can leave the information request there for now.",
            final_outcome="information_cancelled",
        )
    return set_menu(
        state,
        menu_name="change_information",
        menu_options=["change_information", "cancel"],
        response_text=f"{CHANGE_INFORMATION_PROMPT} Just say Change Information to start again, or say cancel if you'd rather stop here.",
    )
