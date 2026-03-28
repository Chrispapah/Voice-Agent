from __future__ import annotations

import logging
from typing import Optional

from vocode.streaming.agent.abstract_factory import AbstractAgentFactory
from vocode.streaming.agent.base_agent import BaseAgent
from vocode.streaming.models.agent import AgentConfig

from vocode_contact_center.agent import ContactCenterAgent, ContactCenterAgentConfig


class ContactCenterAgentFactory(AbstractAgentFactory):
    def create_agent(
        self, agent_config: AgentConfig, logger: Optional[logging.Logger] = None
    ) -> BaseAgent:
        if isinstance(agent_config, ContactCenterAgentConfig):
            return ContactCenterAgent(agent_config=agent_config)
        raise ValueError(f"Unsupported agent config type: {type(agent_config)!r}")
