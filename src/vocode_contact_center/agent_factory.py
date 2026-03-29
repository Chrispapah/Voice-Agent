from __future__ import annotations

import logging
from typing import Optional

from langchain_core.runnables.base import Runnable
from vocode.streaming.agent.abstract_factory import AbstractAgentFactory
from vocode.streaming.agent.base_agent import BaseAgent
from vocode.streaming.models.agent import AgentConfig

from vocode_contact_center.agent import ContactCenterAgent, ContactCenterAgentConfig


class ContactCenterAgentFactory(AbstractAgentFactory):
    def __init__(self, shared_chain: Runnable | None = None):
        self.shared_chain = shared_chain

    def create_agent(
        self, agent_config: AgentConfig, logger: Optional[logging.Logger] = None
    ) -> BaseAgent:
        if isinstance(agent_config, ContactCenterAgentConfig):
            return ContactCenterAgent(
                agent_config=agent_config,
                shared_chain=self.shared_chain,
            )
        raise ValueError(f"Unsupported agent config type: {type(agent_config)!r}")
