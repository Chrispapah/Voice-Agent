from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from loguru import logger

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:  # pragma: no cover
    ChatAnthropic = None

try:
    from langchain_groq import ChatGroq
except ImportError:  # pragma: no cover
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

    async def extract_qualification(
        self,
        *,
        transcript: list[dict[str, str]],
        existing_pain_points: list[str],
    ) -> dict[str, Any]:
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
        if "wrap up" in system_prompt.lower() or "wrapping up" in system_prompt.lower():
            if "booked" in human_text.lower():
                return "Perfect, you are all set. I will send a confirmation email right after this call."
            return "Thanks for your time. I will send a brief follow-up email so you have the next steps in writing."
        return (
            "We help teams automate outbound follow-up so prospects get contacted quickly and "
            "reps spend more time in qualified conversations. Would you be open to a short walkthrough?"
        )

    _EXIT_PHRASES = (
        "not interested", "stop calling", "remove me", "do not call",
        "don't call", "goodbye", "good bye", "bye bye", "hang up",
        "end the call", "end call", "leave me alone", "go away", "get lost",
        "piss off", "fuck off", "fuck you", "screw you", "shut up",
        "stop it", "i'm done", "i am done", "let me go",
        "no thank you", "no thanks", "no thankyou", "nah", "nope",
    )

    _NEGATIVE_PHRASES = (
        "no", "nah", "nope", "not really", "i don't think so",
        "not at this time", "not right now", "i'm good",
    )

    async def classify(
        self,
        *,
        instruction: str,
        human_input: str,
        labels: Sequence[str],
    ) -> str:
        text = human_input.lower().strip()
        if any(phrase in text for phrase in self._EXIT_PHRASES):
            return "wrap_up" if "wrap_up" in labels else "not_interested"
        if "continue_qualifying" in labels:
            qual_signals = ("i handle", "i'm the", "i am the", "i own", "budget",
                            "approved", "quarter", "month", "pain", "problem",
                            "struggle", "slow", "manual")
            if any(s in text for s in qual_signals):
                return "continue_qualifying"
        if "busy" in text or "already have" in text or "send info" in text or "maybe later" in text:
            if "handle_objection" in labels:
                return "handle_objection"
        if "yes" in text or "sounds good" in text or "let's do it" in text:
            if "book_meeting" in labels:
                return "book_meeting"
            if "continue_qualifying" in labels:
                return "continue_qualifying"
            if "pitch" in labels:
                return "pitch"
            if "continue_booking" in labels:
                return "continue_booking"
        if "tuesday" in text or "tomorrow" in text or "3 pm" in text or "10 am" in text:
            if "continue_booking" in labels:
                return "continue_booking"
            if "wrap_up" in labels:
                return "wrap_up"
        if text in self._NEGATIVE_PHRASES or any(text.startswith(p) for p in self._NEGATIVE_PHRASES):
            if "not_interested" in labels:
                return "not_interested"
            if "wrap_up" in labels:
                return "wrap_up"
            if "handle_objection" in labels:
                return "handle_objection"
        return labels[0]

    async def extract_qualification(
        self,
        *,
        transcript: list[dict[str, str]],
        existing_pain_points: list[str],
    ) -> dict[str, Any]:
        text = self._last_human_text(transcript).lower().strip()
        updates: dict[str, Any] = {}
        if any(p in text for p in ("i handle", "i'm the", "i am the", "i own", "my call")):
            updates["is_decision_maker"] = True
        if "budget" in text or "approved" in text:
            updates["budget_confirmed"] = True
        if "quarter" in text or "month" in text:
            updates["timeline"] = self._last_human_text(transcript)
        return updates


class LangChainConversationBrain:
    """LLM-backed brain. Accepts either SDRSettings (legacy) or a bot_config dict."""

    def __init__(self, settings: SDRSettings | None = None, *, bot_config: dict | None = None):
        if bot_config:
            provider = bot_config.get("llm_provider", "openai")
            model_name = bot_config.get("llm_model_name", "gpt-4o-mini")
            temperature = bot_config.get("llm_temperature", 0.4)
            max_tokens = bot_config.get("llm_max_tokens", 300)
            openai_key = bot_config.get("openai_api_key")
            anthropic_key = bot_config.get("anthropic_api_key")
            groq_key = bot_config.get("groq_api_key")
        elif settings:
            provider = settings.llm_provider
            model_name = settings.llm_model_name
            temperature = settings.llm_temperature
            max_tokens = settings.llm_max_tokens
            openai_key = settings.openai_api_key
            anthropic_key = settings.anthropic_api_key
            groq_key = settings.groq_api_key
        else:
            raise ValueError("Either settings or bot_config must be provided")

        if provider == "openai":
            self._model = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=openai_key,
            )
        elif provider == "groq" and ChatGroq is not None:
            self._model = ChatGroq(
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=groq_key,
            )
        elif provider == "anthropic" and ChatAnthropic is not None:
            self._model = ChatAnthropic(
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=anthropic_key,
            )
        else:
            raise ValueError(f"Unsupported llm_provider: {provider}")

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
                        + "\n\nRespond with ONLY the label, nothing else."
                    )
                ),
                HumanMessage(content=human_input),
            ]
        )
        raw = str(response.content).strip().lower()
        raw = raw.strip("'\"`.").strip()
        if raw in labels:
            return raw
        for label in labels:
            if label in raw:
                return label
        return labels[0]

    async def extract_qualification(
        self,
        *,
        transcript: list[dict[str, str]],
        existing_pain_points: list[str],
    ) -> dict[str, Any]:
        from ai_sdr_agent.graph.prompts import qualification_extraction_prompt

        system = qualification_extraction_prompt(existing_pain_points)
        messages = [SystemMessage(content=system)]
        for item in transcript:
            if item["role"] == "human":
                messages.append(HumanMessage(content=item["content"]))
            else:
                messages.append(SystemMessage(content=f"Agent previously said: {item['content']}"))

        response = await self._model.ainvoke(messages)
        raw = str(response.content).strip()

        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Failed to parse qualification JSON, falling back to empty: {!r}", raw)
            return {}

        updates: dict[str, Any] = {}
        if data.get("is_decision_maker") is not None:
            updates["is_decision_maker"] = bool(data["is_decision_maker"])
        if data.get("budget_confirmed") is not None:
            updates["budget_confirmed"] = bool(data["budget_confirmed"])
        if data.get("timeline") is not None:
            updates["timeline"] = str(data["timeline"])
        new_pain = data.get("pain_points") or []
        if isinstance(new_pain, list) and new_pain:
            combined = list(existing_pain_points) + [
                p for p in new_pain if isinstance(p, str) and p not in existing_pain_points
            ]
            updates["pain_points"] = combined
        return updates


def build_conversation_brain(
    settings: SDRSettings | None = None,
    *,
    bot_config: dict | None = None,
) -> ConversationBrain:
    """Build a brain from SDRSettings (legacy) or a per-bot config dict."""
    provider = (
        (bot_config or {}).get("llm_provider")
        or (settings.llm_provider if settings else "stub")
    )
    if provider == "stub":
        return StubConversationBrain()
    return LangChainConversationBrain(settings, bot_config=bot_config)
