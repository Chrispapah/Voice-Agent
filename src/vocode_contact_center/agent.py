from __future__ import annotations

from typing import AsyncGenerator, Optional

from loguru import logger
from langchain_core.runnables.base import Runnable
from pydantic.v1 import Field
from vocode.streaming.agent.base_agent import GeneratedResponse, RespondAgent, StreamedResponse
from vocode.streaming.agent.streaming_utils import collate_response_async, stream_response_async
from vocode.streaming.models.agent import AgentConfig
from vocode.streaming.models.message import BaseMessage, LLMToken

from vocode_contact_center.latency_tracker import conversation_latency_tracker
from vocode_contact_center.langchain_support import (
    astream_text_tokens,
    build_prompt_inputs,
    build_chain,
    extract_text_from_langchain_message,
)


class ContactCenterAgentConfig(AgentConfig, type="agent_contact_center"):  # type: ignore[misc]
    prompt_preamble: str
    model_name: str
    provider: str
    temperature: float = 0.2
    max_tokens: int = 256
    call_context: str = "Live call metadata is unavailable."
    require_streaming_synthesizer: bool = True
    recent_message_limit: int = 6
    summary_max_messages: int = 12
    summary_max_chars: int = 600
    transfer_phone_number: str | None = None
    escalation_keywords: list[str] = Field(
        default_factory=lambda: ["human", "agent", "representative", "manager"]
    )
    fallback_handoff_message: str = (
        "I can take your details and make sure a human agent follows up with you."
    )


class ContactCenterAgent(RespondAgent[ContactCenterAgentConfig]):
    def __init__(
        self,
        agent_config: ContactCenterAgentConfig,
        *,
        shared_chain: Runnable | None = None,
        **kwargs,
    ):
        super().__init__(agent_config=agent_config, **kwargs)
        self.chain = shared_chain or build_chain(agent_config)
        self.call_context = agent_config.call_context
        self._logged_streaming_synthesizer_mode = False

    def _get_call_context(self) -> str:
        return self.call_context or "Live call metadata is unavailable."

    def _build_prompt_inputs(self) -> dict[str, str | list[tuple[str, str]]]:
        if self.transcript is None:
            raise ValueError("A transcript is required before generating responses.")
        return build_prompt_inputs(
            self.transcript,
            call_context=self._get_call_context(),
            recent_message_limit=self.agent_config.recent_message_limit,
            summary_max_messages=self.agent_config.summary_max_messages,
            summary_max_chars=self.agent_config.summary_max_chars,
        )

    def _using_input_streaming_synthesizer(self) -> bool:
        using_input_streaming_synthesizer = (
            hasattr(self, "conversation_state_manager")
            and self.conversation_state_manager.using_input_streaming_synthesizer()
        )
        if not self._logged_streaming_synthesizer_mode:
            log_method = logger.info if using_input_streaming_synthesizer else logger.warning
            log_method(
                "Synthesizer mode for this conversation uses input streaming: {}",
                using_input_streaming_synthesizer,
            )
            self._logged_streaming_synthesizer_mode = True
        if (
            not using_input_streaming_synthesizer
            and self.agent_config.require_streaming_synthesizer
        ):
            raise RuntimeError(
                "This deployment requires an input-streaming synthesizer to avoid "
                "response collation on live calls. Disable "
                "REQUIRE_STREAMING_SYNTHESIZER to allow the slower fallback."
            )
        return using_input_streaming_synthesizer

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

        prompt_inputs = self._build_prompt_inputs()
        result = await self.chain.ainvoke(prompt_inputs)
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

        using_input_streaming_synthesizer = self._using_input_streaming_synthesizer()
        prompt_inputs = self._build_prompt_inputs()
        stream = self.chain.astream(prompt_inputs)
        response_generator = (
            stream_response_async if using_input_streaming_synthesizer else collate_response_async
        )

        first_token_logged = False
        async for message in response_generator(
            conversation_id=conversation_id,
            gen=astream_text_tokens(stream),
        ):
            response_class = StreamedResponse if using_input_streaming_synthesizer else GeneratedResponse
            message_type = LLMToken if using_input_streaming_synthesizer else BaseMessage

            if isinstance(message, str):
                if not first_token_logged:
                    conversation_latency_tracker.mark_first_llm_token(
                        conversation_id,
                        using_input_streaming_synthesizer=using_input_streaming_synthesizer,
                    )
                    first_token_logged = True
                yield response_class(
                    message=message_type(text=message),
                    is_interruptible=True,
                )
            else:
                logger.warning("Skipping unsupported non-text message from custom LangChain agent.")


def build_call_context(*, from_phone: str | None, to_phone: str | None) -> str:
    context_parts = ["Live call metadata:"]
    if from_phone:
        context_parts.append(f"- Caller number: {from_phone}")
    if to_phone:
        context_parts.append(f"- Dialed number: {to_phone}")
    if len(context_parts) == 1:
        context_parts.append("- Caller metadata was not available during call setup.")
    return "\n".join(context_parts)
