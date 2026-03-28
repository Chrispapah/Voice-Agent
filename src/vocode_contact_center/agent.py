from __future__ import annotations

from typing import AsyncGenerator, Optional

from loguru import logger
from pydantic.v1 import Field
from vocode.streaming.agent.base_agent import GeneratedResponse, RespondAgent, StreamedResponse
from vocode.streaming.agent.streaming_utils import collate_response_async, stream_response_async
from vocode.streaming.models.agent import AgentConfig
from vocode.streaming.models.message import BaseMessage, LLMToken
from vocode.streaming.telephony.config_manager.redis_config_manager import RedisConfigManager

from vocode_contact_center.langchain_support import (
    astream_text_tokens,
    build_chain,
    extract_text_from_langchain_message,
    transcript_to_langchain_messages,
)


class ContactCenterAgentConfig(AgentConfig, type="agent_contact_center"):  # type: ignore[misc]
    prompt_preamble: str
    model_name: str
    provider: str
    temperature: float = 0.2
    max_tokens: int = 256
    transfer_phone_number: str | None = None
    escalation_keywords: list[str] = Field(
        default_factory=lambda: ["human", "agent", "representative", "manager"]
    )
    fallback_handoff_message: str = (
        "I can take your details and make sure a human agent follows up with you."
    )


class ContactCenterAgent(RespondAgent[ContactCenterAgentConfig]):
    def __init__(self, agent_config: ContactCenterAgentConfig, **kwargs):
        super().__init__(agent_config=agent_config, **kwargs)
        self.chain = build_chain(agent_config)
        self.config_manager = RedisConfigManager()

    async def _get_call_context(self, conversation_id: str) -> str:
        call_config = await self.config_manager.get_config(conversation_id)
        if call_config is None:
            return "Call metadata is unavailable."

        from_phone = getattr(call_config, "from_phone", None)
        to_phone = getattr(call_config, "to_phone", None)
        context_parts = ["Live call metadata:"]
        if from_phone:
            context_parts.append(f"- Caller number: {from_phone}")
        if to_phone:
            context_parts.append(f"- Dialed number: {to_phone}")
        return "\n".join(context_parts)

    def _looks_like_handoff_request(self, human_input: str) -> bool:
        normalized = human_input.lower()
        return any(keyword in normalized for keyword in self.agent_config.escalation_keywords)

    async def respond(
        self,
        human_input: str,
        conversation_id: str,
        is_interrupt: bool = False,
    ) -> tuple[Optional[str], bool]:
        if self._looks_like_handoff_request(human_input):
            return self.agent_config.fallback_handoff_message, False

        if self.transcript is None:
            raise ValueError("A transcript is required before generating responses.")

        result = await self.chain.ainvoke(
            {
                "chat_history": transcript_to_langchain_messages(self.transcript),
                "call_context": await self._get_call_context(conversation_id),
            }
        )
        return extract_text_from_langchain_message(result), False

    async def generate_response(
        self,
        human_input: str,
        conversation_id: str,
        is_interrupt: bool = False,
        bot_was_in_medias_res: bool = False,
    ) -> AsyncGenerator[GeneratedResponse, None]:
        if self._looks_like_handoff_request(human_input):
            yield GeneratedResponse(
                message=BaseMessage(text=self.agent_config.fallback_handoff_message),
                is_interruptible=True,
            )
            return

        if self.transcript is None:
            raise ValueError("A transcript is required before generating responses.")

        stream = self.chain.astream(
            {
                "chat_history": transcript_to_langchain_messages(self.transcript),
                "call_context": await self._get_call_context(conversation_id),
            }
        )

        response_generator = collate_response_async
        using_input_streaming_synthesizer = (
            hasattr(self, "conversation_state_manager")
            and self.conversation_state_manager.using_input_streaming_synthesizer()
        )
        if using_input_streaming_synthesizer:
            response_generator = stream_response_async

        async for message in response_generator(
            conversation_id=conversation_id,
            gen=astream_text_tokens(stream),
        ):
            response_class = StreamedResponse if using_input_streaming_synthesizer else GeneratedResponse
            message_type = LLMToken if using_input_streaming_synthesizer else BaseMessage

            if isinstance(message, str):
                yield response_class(
                    message=message_type(text=message),
                    is_interruptible=True,
                )
            else:
                logger.warning("Skipping unsupported non-text message from custom LangChain agent.")
