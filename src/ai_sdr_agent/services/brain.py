from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:  # pragma: no cover - optional dependency in local dev
    ChatAnthropic = None

try:
    from langchain_groq import ChatGroq
except ImportError:  # pragma: no cover - optional dependency in local dev
    ChatGroq = None

from ai_sdr_agent.config import SDRSettings


class ConversationBrain(Protocol):
    async def respond(self, *, system_prompt: str, transcript: list[dict[str, str]]) -> str:
        ...

    async def classify(
        self,
        *,
        instruction: str,
        human_input: str,
        labels: Sequence[str],
    ) -> str:
        ...


@dataclass
class StubConversationBrain:
    def _last_human_text(self, transcript: list[dict[str, str]]) -> str:
        for message in reversed(transcript):
            if message["role"] == "human":
                return message["content"]
        return ""

    async def respond(self, *, system_prompt: str, transcript: list[dict[str, str]]) -> str:
        human_text = self._last_human_text(transcript).strip()
        if "greet them warmly" in system_prompt.lower():
            return "Hi, this is Ava following up on your recent interest. Did I catch you at an okay time?"
        if "qualify the prospect" in system_prompt.lower():
            return (
                "Thanks. To make sure this is relevant, are you the person who owns sales process decisions "
                "and are you looking to improve follow-up speed this quarter?"
            )
        if "objection" in system_prompt.lower():
            return (
                "That makes sense. Teams usually start small with one workflow, so the first step is low lift. "
                "Would it help if I showed you what that looks like in a short demo?"
            )
        if "book a meeting" in system_prompt.lower():
            return (
                "I can get something on the calendar. I have tomorrow at 3 PM UTC, "
                "two days from now at 10 AM UTC, or three days from now at 5 PM UTC."
            )
        if "wrap up" in system_prompt.lower():
            if "booked" in human_text.lower():
                return "Perfect, you are all set. I will send a confirmation email right after this call."
            return "Thanks for your time. I will send a brief follow-up email so you have the next steps in writing."
        return (
            "We help teams automate outbound follow-up so prospects get contacted quickly and "
            "reps spend more time in qualified conversations. Would you be open to a short walkthrough?"
        )

    async def classify(
        self,
        *,
        instruction: str,
        human_input: str,
        labels: Sequence[str],
    ) -> str:
        text = human_input.lower()
        if "not interested" in text or "stop calling" in text or "remove me" in text:
            return "wrap_up" if "wrap_up" in labels else "not_interested"
        if "busy" in text or "already have" in text or "send info" in text or "maybe later" in text:
            if "handle_objection" in labels:
                return "handle_objection"
        if "yes" in text or "sounds good" in text or "let's do it" in text:
            if "book_meeting" in labels:
                return "book_meeting"
            if "pitch" in labels:
                return "pitch"
        if "tuesday" in text or "tomorrow" in text or "3 pm" in text or "10 am" in text:
            if "wrap_up" in labels:
                return "wrap_up"
        return labels[0]


class LangChainConversationBrain:
    def __init__(self, settings: SDRSettings):
        self.settings = settings
        if settings.llm_provider == "openai":
            self._model = ChatOpenAI(
                model=settings.llm_model_name,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
                api_key=settings.openai_api_key,
            )
        elif settings.llm_provider == "groq" and ChatGroq is not None:
            self._model = ChatGroq(
                model=settings.llm_model_name,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
                api_key=settings.groq_api_key,
            )
        elif settings.llm_provider == "anthropic" and ChatAnthropic is not None:
            self._model = ChatAnthropic(
                model=settings.llm_model_name,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
                api_key=settings.anthropic_api_key,
            )
        else:
            raise ValueError(f"Unsupported llm_provider: {settings.llm_provider}")

    async def respond(self, *, system_prompt: str, transcript: list[dict[str, str]]) -> str:
        messages = [SystemMessage(content=system_prompt)]
        for item in transcript:
            if item["role"] == "human":
                messages.append(HumanMessage(content=item["content"]))
            else:
                messages.append(SystemMessage(content=f"Agent previously said: {item['content']}"))
        response = await self._model.ainvoke(messages)
        return str(response.content)

    async def classify(
        self,
        *,
        instruction: str,
        human_input: str,
        labels: Sequence[str],
    ) -> str:
        response = await self._model.ainvoke(
            [
                SystemMessage(
                    content=(
                        f"{instruction}\n\nReturn exactly one of these labels: "
                        + ", ".join(labels)
                    )
                ),
                HumanMessage(content=human_input),
            ]
        )
        label = str(response.content).strip().lower()
        if label not in labels:
            return labels[0]
        return label


def build_conversation_brain(settings: SDRSettings) -> ConversationBrain:
    if settings.llm_provider == "stub":
        return StubConversationBrain()
    return LangChainConversationBrain(settings)
