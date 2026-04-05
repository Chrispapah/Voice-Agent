from __future__ import annotations

import asyncio
from collections.abc import Callable

from langgraph.graph import END, START, StateGraph
from loguru import logger

from vocode_contact_center.settings import ContactCenterSettings
from vocode_contact_center.product_knowledge import ProductKnowledgeService
from vocode_contact_center.voicebot_graph.adapters.base import (
    AuthenticationAdapter,
    GenesysAdapter,
    SmsRequest,
    SmsSender,
)
from vocode_contact_center.voicebot_graph.nodes import announcements, feedback, information, interaction, root
from vocode_contact_center.voicebot_graph.state import VoicebotGraphState


def build_voicebot_graph(
    *,
    settings: ContactCenterSettings,
    product_knowledge: ProductKnowledgeService,
    auth_adapter: AuthenticationAdapter,
    genesys_adapter: GenesysAdapter,
    sms_sender: SmsSender,
    schedule_background_sms: Callable[[SmsRequest], None] | None = None,
):
    auth_timeout = max(1.0, float(settings.adapter_authentication_timeout_seconds))
    genesys_timeout = max(1.0, float(settings.adapter_genesys_timeout_seconds))
    defer_sms = bool(
        settings.defer_sms_send_in_background and schedule_background_sms is not None
    )

    async def run_authentication(state: VoicebotGraphState) -> VoicebotGraphState:
        try:
            return await asyncio.wait_for(
                interaction.authenticate(state, auth_adapter),
                timeout=auth_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Authentication adapter timed out session={}",
                state.get("session_id"),
            )
            return {
                "auth_status": "failure",
                "terminal_group": "fail_terminal",
                "route_decision": "interaction_terminal",
                "response_text": (
                    "I couldn't reach the verification service in time. "
                    "We can try again in a moment, or I can help with something else."
                ),
                "pending_prompt": (
                    "I couldn't reach the verification service in time. "
                    "We can try again in a moment, or I can help with something else."
                ),
            }

    async def run_announcements_genesys(state: VoicebotGraphState) -> VoicebotGraphState:
        try:
            return await asyncio.wait_for(
                announcements.call_genesys(state, genesys_adapter),
                timeout=genesys_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Genesys announcements connect timed out session={}",
                state.get("session_id"),
            )
            return {
                "response_text": (
                    "I couldn't reach the contact center routing system in time. "
                    "Would you like to try again or choose another option?"
                ),
                "pending_prompt": (
                    "I couldn't reach the contact center routing system in time. "
                    "Would you like to try again or choose another option?"
                ),
                "route_decision": "announcements",
                "genesys_requested": False,
            }

    async def run_feedback_genesys(state: VoicebotGraphState) -> VoicebotGraphState:
        try:
            return await asyncio.wait_for(
                feedback.call_genesys(state, genesys_adapter),
                timeout=genesys_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Genesys feedback connect timed out session={}",
                state.get("session_id"),
            )
            return {
                "response_text": (
                    "I couldn't reach the contact center routing system in time. "
                    "Would you like to try again or pick a different option?"
                ),
                "pending_prompt": (
                    "I couldn't reach the contact center routing system in time. "
                    "Would you like to try again or pick a different option?"
                ),
                "route_decision": "feedback",
                "genesys_requested": False,
            }

    async def run_sms_confirmation(state: VoicebotGraphState) -> VoicebotGraphState:
        return await interaction.sms_confirmation(
            state,
            sms_sender,
            settings,
            defer_sms=defer_sms,
            schedule_background_sms=schedule_background_sms if defer_sms else None,
        )

    async def run_interaction_terminal(state: VoicebotGraphState) -> VoicebotGraphState:
        return await interaction.handle_terminal_menu(
            state,
            sms_sender,
            settings,
            defer_sms=defer_sms,
            schedule_background_sms=schedule_background_sms if defer_sms else None,
        )

    async def run_information(state: VoicebotGraphState) -> VoicebotGraphState:
        return await information.handle_information(state, settings, product_knowledge)

    builder = StateGraph(VoicebotGraphState)

    builder.add_node("route_turn", root.route_turn)
    builder.add_node("root_intent_node", root.resolve_root_intent)
    builder.add_node("global_main_menu", root.return_to_main_menu)
    builder.add_node("information", run_information)
    builder.add_node("interaction_entry", interaction.handle_interaction_entry)
    builder.add_node("interaction_authenticate", run_authentication)
    builder.add_node("interaction_customer_input", interaction.collect_customer_input)
    builder.add_node("interaction_sms_confirmation", run_sms_confirmation)
    builder.add_node("interaction_terminal", run_interaction_terminal)
    builder.add_node("announcements", lambda state: announcements.handle_announcements(state, settings))
    builder.add_node("announcements_genesys", run_announcements_genesys)
    builder.add_node("feedback", lambda state: feedback.handle_feedback(state, settings))
    builder.add_node("feedback_genesys", run_feedback_genesys)
    builder.add_node("complete", lambda state: {})

    builder.add_edge(START, "route_turn")

    route_map = {
        "root_intent": "root_intent_node",
        "global_main_menu": "global_main_menu",
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
        "global_main_menu",
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
