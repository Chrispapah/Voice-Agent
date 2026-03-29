from __future__ import annotations

from functools import lru_cache
from typing import Any, AsyncGenerator

from langchain.chat_models import init_chat_model
from langchain_core.messages.base import BaseMessage as LangChainBaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.base import Runnable
from vocode.streaming.models.events import Sender
from vocode.streaming.models.transcript import Message, Transcript

def build_chain(agent_config: Any) -> Runnable:
    return _build_cached_chain(
        prompt_preamble=agent_config.prompt_preamble,
        model_name=agent_config.model_name,
        provider=agent_config.provider,
        temperature=agent_config.temperature,
        max_tokens=agent_config.max_tokens,
    )


@lru_cache(maxsize=16)
def _build_cached_chain(
    *,
    prompt_preamble: str,
    model_name: str,
    provider: str,
    temperature: float,
    max_tokens: int,
) -> Runnable:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt_preamble),
            ("system", "{call_context}"),
            ("system", "{conversation_summary}"),
            ("placeholder", "{chat_history}"),
        ]
    )
    model = init_chat_model(
        model=model_name,
        model_provider=provider,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return prompt | model


def transcript_to_langchain_messages(
    transcript: Transcript,
    *,
    message_limit: int | None = None,
) -> list[tuple[str, str]]:
    messages: list[tuple[str, str]] = []
    for event_log in transcript.event_logs:
        if isinstance(event_log, Message):
            role = "ai" if event_log.sender == Sender.BOT else "human"
            messages.append((role, event_log.to_string(include_sender=False)))
    if message_limit is not None and message_limit > 0:
        return messages[-message_limit:]
    return messages


def build_prompt_inputs(
    transcript: Transcript,
    *,
    call_context: str,
    recent_message_limit: int,
    summary_max_messages: int,
    summary_max_chars: int,
) -> dict[str, str | list[tuple[str, str]]]:
    return build_prompt_inputs_from_messages(
        transcript_to_langchain_messages(transcript),
        call_context=call_context,
        recent_message_limit=recent_message_limit,
        summary_max_messages=summary_max_messages,
        summary_max_chars=summary_max_chars,
    )


def build_prompt_inputs_from_messages(
    messages: list[tuple[str, str]],
    *,
    call_context: str,
    recent_message_limit: int,
    summary_max_messages: int,
    summary_max_chars: int,
) -> dict[str, str | list[tuple[str, str]]]:
    all_messages = messages
    recent_limit = max(recent_message_limit, 1)
    recent_messages = all_messages[-recent_limit:]
    older_messages = all_messages[:-recent_limit]
    return {
        "call_context": call_context,
        "conversation_summary": summarize_messages(
            older_messages,
            max_messages=summary_max_messages,
            max_chars=summary_max_chars,
        ),
        "chat_history": recent_messages,
    }


def summarize_messages(
    messages: list[tuple[str, str]],
    *,
    max_messages: int,
    max_chars: int,
) -> str:
    if not messages or max_messages <= 0 or max_chars <= 0:
        return "Conversation summary: no earlier turns."

    clipped_messages = messages[-max_messages:]
    parts: list[str] = []
    char_budget = max_chars
    omitted_count = max(0, len(messages) - len(clipped_messages))

    for role, text in clipped_messages:
        normalized = " ".join(text.split())
        if not normalized:
            continue
        speaker = "Agent" if role == "ai" else "Caller"
        snippet = normalized[:160].rstrip()
        if len(normalized) > 160:
            snippet = f"{snippet}..."
        part = f"{speaker}: {snippet}"
        separator = " | " if parts else ""
        required_chars = len(separator) + len(part)
        if required_chars > char_budget:
            break
        parts.append(f"{separator}{part}" if separator else part)
        char_budget -= required_chars

    if not parts:
        return "Conversation summary: earlier turns exist but exceeded the summary budget."

    omitted_prefix = ""
    if omitted_count:
        omitted_prefix = f"Conversation summary: {omitted_count} earlier turns compressed. "
    else:
        omitted_prefix = "Conversation summary: "
    return omitted_prefix + "".join(parts)


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
