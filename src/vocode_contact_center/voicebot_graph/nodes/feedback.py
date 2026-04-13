from __future__ import annotations

from vocode_contact_center.settings import ContactCenterSettings
from vocode_contact_center.voicebot_graph.adapters.base import GenesysAdapter, GenesysRequest
from vocode_contact_center.voicebot_graph.intents import (
    classify_feedback_question_choice,
    classify_terminal_choice,
)
from vocode_contact_center.voicebot_graph.nodes.terminals import (
    complete_path,
    set_menu,
    terminal_response_text,
)
from vocode_contact_center.voicebot_graph.state import VoicebotGraphState


def handle_feedback(
    state: VoicebotGraphState,
    settings: ContactCenterSettings,
) -> VoicebotGraphState:
    active_menu = state.get("active_menu")
    if active_menu == "feedback_terminal":
        choice = classify_terminal_choice(state.get("latest_user_input", ""), "feedback_terminal")
        if choice is None:
            return _prompt_feedback_terminal()
        return complete_path(
            state,
            response_text=terminal_response_text("feedback_terminal", choice),
            final_outcome=choice,
        )

    if active_menu == "feedback_question":
        choice = classify_feedback_question_choice(state.get("latest_user_input", ""))
        if choice == "back_to_chat":
            return complete_path(
                state,
                response_text="Absolutely. I'll send you back to the chat now.",
                final_outcome="back_to_chat",
            )
        if choice == "genesys":
            return {"route_decision": "feedback_genesys"}
        return _prompt_feedback_question(settings)

    return _prompt_feedback_question(settings)


async def call_genesys(state: VoicebotGraphState, genesys_adapter: GenesysAdapter) -> VoicebotGraphState:
    result = await genesys_adapter.connect(
        GenesysRequest(
            session_id=state["session_id"],
            path_name="feedback",
            latest_user_input=state.get("latest_user_input", ""),
            metadata=dict(state.get("metadata", {})),
        )
    )
    return {
        "genesys_requested": True,
        "active_menu": "feedback_terminal",
        "menu_options": ["human_agent", "contact"],
        "adapter_results": {
            "genesys": {
                "status": result.status,
                "metadata": result.metadata,
            }
        },
        "route_decision": "feedback",
    }


def _prompt_feedback_question(settings: ContactCenterSettings) -> VoicebotGraphState:
    return set_menu(
        {},
        menu_name="feedback_question",
        menu_options=["back_to_chat", "genesys"],
        response_text=settings.feedback_question_prompt,
    )


def _prompt_feedback_terminal() -> VoicebotGraphState:
    return set_menu(
        {},
        menu_name="feedback_terminal",
        menu_options=["human_agent", "contact"],
        response_text="I can connect you with a human agent, or I can submit a contact request for follow-up. Which would you prefer?",
    )
