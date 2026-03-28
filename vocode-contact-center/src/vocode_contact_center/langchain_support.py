from __future__ import annotations

from typing import Any, AsyncGenerator

from langchain.chat_models import init_chat_model
from langchain_core.messages.base import BaseMessage as LangChainBaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.base import Runnable
from vocode.streaming.models.events import Sender
from vocode.streaming.models.transcript import Message, Transcript

def build_chain(agent_config: Any) -> Runnable:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", agent_config.prompt_preamble),
            ("system", "{call_context}"),
            ("placeholder", "{chat_history}"),
        ]
    )
    model = init_chat_model(
        model=agent_config.model_name,
        model_provider=agent_config.provider,
        temperature=agent_config.temperature,
        max_tokens=agent_config.max_tokens,
    )
    return prompt | model


def transcript_to_langchain_messages(transcript: Transcript) -> list[tuple[str, str]]:
    messages: list[tuple[str, str]] = []
    for event_log in transcript.event_logs:
        if isinstance(event_log, Message):
            role = "ai" if event_log.sender == Sender.BOT else "human"
            messages.append((role, event_log.to_string(include_sender=False)))
    return messages


def extract_text_from_langchain_message(message: LangChainBaseMessage) -> str:
    if isinstance(message.content, str):
        return message.content

    parts: list[str] = []
    for chunk in message.content:
        if isinstance(chunk, str):
            parts.append(chunk)
            continue
        if isinstance(chunk, dict) and chunk.get("type") == "text":
            text = chunk.get("text")
            if text:
                parts.append(text)
    return "".join(parts).strip()


async def astream_text_tokens(stream: AsyncGenerator[LangChainBaseMessage, None]):
    async for chunk in stream:
        text = extract_text_from_langchain_message(chunk)
        if text:
            yield text
