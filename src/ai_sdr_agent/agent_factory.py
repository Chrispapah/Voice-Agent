from __future__ import annotations

import logging
from typing import Optional

from vocode.streaming.agent.abstract_factory import AbstractAgentFactory
from vocode.streaming.agent.base_agent import BaseAgent
from vocode.streaming.models.agent import AgentConfig

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
        if isinstance(agent_config, SDRAgentConfig):
            return SDRVocodeAgent(
                agent_config=agent_config,
                conversation_service=self.conversation_service,
            )
        raise ValueError(f"Unsupported agent config type: {type(agent_config)!r}")
