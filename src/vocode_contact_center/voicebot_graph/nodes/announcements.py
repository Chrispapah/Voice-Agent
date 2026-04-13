from __future__ import annotations

from vocode_contact_center.settings import ContactCenterSettings
from vocode_contact_center.voicebot_graph.adapters.base import GenesysAdapter, GenesysRequest
from vocode_contact_center.voicebot_graph.intents import (
    classify_announcements_continue_choice,
    classify_terminal_choice,
)
from vocode_contact_center.voicebot_graph.nodes.terminals import (
    complete_path,
    set_menu,
    terminal_response_text,
)
from vocode_contact_center.voicebot_graph.state import VoicebotGraphState


def handle_announcements(
    state: VoicebotGraphState,
    settings: ContactCenterSettings,
) -> VoicebotGraphState:
    active_menu = state.get("active_menu")
    if active_menu == "announcements_terminal":
        choice = classify_terminal_choice(state.get("latest_user_input", ""), "announcements_terminal")
        if choice is None:
            return _prompt_terminal_menu()
        return complete_path(
            state,
            response_text=terminal_response_text("announcements_terminal", choice),
            final_outcome=choice,
        )

    if active_menu == "announcements_continue":
        choice = classify_announcements_continue_choice(state.get("latest_user_input", ""))
        if choice == "continue":
            return {"route_decision": "announcements_genesys"}
        if choice == "stop":
            return complete_path(
                state,
                response_text="Of course. I'll leave it there after the announcements.",
                final_outcome="announcements_stopped",
            )
        return _prompt_continue(settings)

    return _prompt_continue(settings)


async def call_genesys(state: VoicebotGraphState, genesys_adapter: GenesysAdapter) -> VoicebotGraphState:
    result = await genesys_adapter.connect(
        GenesysRequest(
            session_id=state["session_id"],
            path_name="announcements",
            latest_user_input=state.get("latest_user_input", ""),
            metadata=dict(state.get("metadata", {})),
        )
    )
    return {
        "genesys_requested": True,
        "active_menu": "announcements_terminal",
        "menu_options": ["human_agent", "call_back"],
        "adapter_results": {
            "genesys": {
                "status": result.status,
                "metadata": result.metadata,
            }
        },
        "route_decision": "announcements",
    }


def _prompt_continue(settings: ContactCenterSettings) -> VoicebotGraphState:
    response_text = (
        f"{settings.announcements_message} If you'd like, I can also connect you to contact center support after that. Would you like me to continue?"
    )
    return set_menu(
        {},
        menu_name="announcements_continue",
        menu_options=["continue", "stop"],
        response_text=response_text,
    )


def _prompt_terminal_menu() -> VoicebotGraphState:
    return set_menu(
        {},
        menu_name="announcements_terminal",
        menu_options=["human_agent", "call_back"],
        response_text="I can connect you to a human agent, or I can arrange a call back instead. Which works better for you?",
    )
