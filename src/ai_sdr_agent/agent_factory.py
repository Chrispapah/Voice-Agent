from __future__ import annotations

import logging
from typing import Optional

from vocode.streaming.agent.abstract_factory import AbstractAgentFactory
from vocode.streaming.agent.base_agent import BaseAgent
from vocode.streaming.models.agent import AgentConfig
from vocode.streaming.models.message import BaseMessage

from ai_sdr_agent.graph.service import SDRConversationService
from ai_sdr_agent.vocode_agent import SDRAgentConfig, SDRVocodeAgent


class SDRAgentFactory(AbstractAgentFactory):
    def __init__(self, conversation_service: SDRConversationService):
        self.conversation_service = conversation_service

    def create_agent(
        self,
        agent_config: AgentConfig,
        logger: Optional[logging.Logger] = None,
    ) -> BaseAgent:
        if not isinstance(agent_config, SDRAgentConfig):
            initial_message = getattr(agent_config, "initial_message", None)
            initial_message_text = (
                getattr(initial_message, "text", None)
                if isinstance(initial_message, BaseMessage)
                else None
            ) or "Hi, this is Ava. Is now a good time for a quick chat?"
            agent_config = SDRAgentConfig(
                lead_id=getattr(agent_config, "lead_id", "lead-001"),
                calendar_id=getattr(agent_config, "calendar_id", "sales-team"),
                sales_rep_name=getattr(agent_config, "sales_rep_name", "Sales Team"),
                initial_message_text=getattr(
                    agent_config, "initial_message_text", initial_message_text
                ),
                generate_responses=False,
                allow_agent_to_be_cut_off=getattr(
                    agent_config, "allow_agent_to_be_cut_off", True
                ),
                interrupt_sensitivity=getattr(agent_config, "interrupt_sensitivity", "high"),
                end_conversation_on_goodbye=getattr(
                    agent_config, "end_conversation_on_goodbye", True
                ),
            )
        return SDRVocodeAgent(
            agent_config=agent_config,
            conversation_service=self.conversation_service,
        )
