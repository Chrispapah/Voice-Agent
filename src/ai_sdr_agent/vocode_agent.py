from __future__ import annotations

from loguru import logger
from vocode.streaming.agent.base_agent import RespondAgent
from vocode.streaming.models.agent import AgentConfig
from vocode.streaming.models.message import BaseMessage

from ai_sdr_agent.graph.service import SDRConversationService


class SDRAgentConfig(AgentConfig, type="agent_sdr"):  # type: ignore[misc]
    lead_id: str
    calendar_id: str
    sales_rep_name: str
    initial_message_text: str


class SDRVocodeAgent(RespondAgent[SDRAgentConfig]):
    def __init__(
        self,
        agent_config: SDRAgentConfig,
        *,
        conversation_service: SDRConversationService,
        **kwargs,
    ):
        super().__init__(agent_config=agent_config, **kwargs)
        self.conversation_service = conversation_service
        self._initialized_conversations: set[str] = set()

    async def respond(
        self,
        human_input: str,
        conversation_id: str,
        is_interrupt: bool = False,
    ) -> tuple[str | None, bool]:
        logger.info(
            "SDR agent received speech conversation_id={} is_interrupt={} text={!r}",
            conversation_id,
            is_interrupt,
            human_input,
        )
        if conversation_id not in self._initialized_conversations:
            await self.conversation_service.start_session(
                self.agent_config.lead_id,
                conversation_id=conversation_id,
            )
            self._initialized_conversations.add(conversation_id)
            logger.info(
                "Initialized SDR session from phone call conversation_id={} lead_id={}",
                conversation_id,
                self.agent_config.lead_id,
            )
        state = await self.conversation_service.handle_turn(conversation_id, human_input)
        logger.info(
            "SDR agent generated reply conversation_id={} text={!r}",
            conversation_id,
            state["last_agent_response"],
        )
        return state["last_agent_response"], False


def build_agent_config(
    *,
    lead_id: str,
    calendar_id: str,
    sales_rep_name: str,
    initial_message_text: str,
    allow_agent_to_be_cut_off: bool = True,
    interrupt_sensitivity: str = "high",
) -> SDRAgentConfig:
    return SDRAgentConfig(
        lead_id=lead_id,
        calendar_id=calendar_id,
        sales_rep_name=sales_rep_name,
        initial_message_text=initial_message_text,
        initial_message=BaseMessage(text=initial_message_text),
        # RespondAgent subclasses must use respond(), not generate_response().
        generate_responses=False,
        allow_agent_to_be_cut_off=allow_agent_to_be_cut_off,
        interrupt_sensitivity=interrupt_sensitivity,
        end_conversation_on_goodbye=True,
    )
