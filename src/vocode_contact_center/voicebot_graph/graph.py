from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from vocode_contact_center.settings import ContactCenterSettings
from vocode_contact_center.voicebot_graph.adapters.base import (
    AuthenticationAdapter,
    GenesysAdapter,
    SmsSender,
)
from vocode_contact_center.voicebot_graph.nodes import announcements, feedback, information, interaction, root
from vocode_contact_center.voicebot_graph.state import VoicebotGraphState


def build_voicebot_graph(
    *,
    settings: ContactCenterSettings,
    auth_adapter: AuthenticationAdapter,
    genesys_adapter: GenesysAdapter,
    sms_sender: SmsSender,
):
    async def run_authentication(state: VoicebotGraphState) -> VoicebotGraphState:
        return await interaction.authenticate(state, auth_adapter)

    async def run_announcements_genesys(state: VoicebotGraphState) -> VoicebotGraphState:
        return await announcements.call_genesys(state, genesys_adapter)

    async def run_feedback_genesys(state: VoicebotGraphState) -> VoicebotGraphState:
        return await feedback.call_genesys(state, genesys_adapter)

    async def run_sms_confirmation(state: VoicebotGraphState) -> VoicebotGraphState:
        return await interaction.sms_confirmation(state, sms_sender, settings)

    builder = StateGraph(VoicebotGraphState)

    builder.add_node("route_turn", root.route_turn)
    builder.add_node("root_intent_node", root.resolve_root_intent)
    builder.add_node("information", lambda state: information.handle_information(state, settings))
    builder.add_node("interaction_entry", interaction.handle_interaction_entry)
    builder.add_node("interaction_authenticate", run_authentication)
    builder.add_node("interaction_customer_input", interaction.collect_customer_input)
    builder.add_node("interaction_sms_confirmation", run_sms_confirmation)
    builder.add_node("interaction_terminal", interaction.handle_terminal_menu)
    builder.add_node("announcements", lambda state: announcements.handle_announcements(state, settings))
    builder.add_node("announcements_genesys", run_announcements_genesys)
    builder.add_node("feedback", lambda state: feedback.handle_feedback(state, settings))
    builder.add_node("feedback_genesys", run_feedback_genesys)
    builder.add_node("complete", lambda state: {})

    builder.add_edge(START, "route_turn")

    route_map = {
        "root_intent": "root_intent_node",
        "information": "information",
        "interaction_entry": "interaction_entry",
        "interaction_authenticate": "interaction_authenticate",
        "interaction_customer_input": "interaction_customer_input",
        "interaction_sms_confirmation": "interaction_sms_confirmation",
        "interaction_terminal": "interaction_terminal",
        "announcements": "announcements",
        "announcements_genesys": "announcements_genesys",
        "announcements_terminal": "announcements",
        "feedback": "feedback",
        "feedback_genesys": "feedback_genesys",
        "feedback_terminal": "feedback",
        "complete": "complete",
    }

    def add_routes(node_name: str) -> None:
        builder.add_conditional_edges(node_name, _route_decision, route_map)

    for node_name in (
        "route_turn",
        "root_intent_node",
        "information",
        "interaction_entry",
        "interaction_authenticate",
        "interaction_customer_input",
        "interaction_sms_confirmation",
        "interaction_terminal",
        "announcements",
        "announcements_genesys",
        "feedback",
        "feedback_genesys",
    ):
        add_routes(node_name)

    builder.add_edge("complete", END)

    return builder.compile()


def _route_decision(state: VoicebotGraphState) -> str:
    return state.get("route_decision", "complete")
