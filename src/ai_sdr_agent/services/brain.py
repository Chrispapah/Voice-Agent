from __future__ import annotations

import asyncio
import json
import time
import uuid
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


def _trace_value(trace: dict[str, Any] | None, key: str, default: str = "-") -> str:
    if not trace:
        return default
    value = trace.get(key)
    if value is None:
        return default
    return str(value)


def _preview_text(text: str, limit: int = 80) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _last_human_text(transcript: list[dict[str, str]]) -> str:
    for message in reversed(transcript):
        if message["role"] == "human":
            return message["content"]
    return ""


def _count_role_messages(transcript: list[dict[str, str]], role: str) -> int:
    return sum(1 for message in transcript if message["role"] == role)


def _slice_transcript(
    transcript: list[dict[str, str]],
    *,
    max_messages: int,
) -> list[dict[str, str]]:
    if max_messages <= 0 or len(transcript) <= max_messages:
        return list(transcript)
    return list(transcript[-max_messages:])


def _messages_from_transcript(
    transcript: list[dict[str, str]],
) -> list[HumanMessage | SystemMessage]:
    messages: list[HumanMessage | SystemMessage] = []
    for item in transcript:
        if item["role"] == "human":
            messages.append(HumanMessage(content=item["content"]))
        else:
            messages.append(SystemMessage(content=f"Agent previously said: {item['content']}"))
    return messages


class ConversationBrain(Protocol):
    async def respond(
        self,
        *,
        system_prompt: str,
        transcript: list[dict[str, str]],
        max_tokens: int | None = None,
        trace: dict[str, Any] | None = None,
    ) -> str:
        ...

    async def classify(
        self,
        *,
        instruction: str,
        human_input: str,
        labels: Sequence[str],
        trace: dict[str, Any] | None = None,
    ) -> str:
        ...

    async def extract_qualification(
        self,
        *,
        transcript: list[dict[str, str]],
        existing_pain_points: list[str],
        trace: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        ...


@dataclass
class StubConversationBrain:
    def _last_human_text(self, transcript: list[dict[str, str]]) -> str:
        return _last_human_text(transcript)

    async def respond(
        self,
        *,
        system_prompt: str,
        transcript: list[dict[str, str]],
        max_tokens: int | None = None,
        trace: dict[str, Any] | None = None,
    ) -> str:
        human_text = self._last_human_text(transcript).strip()
        pl = system_prompt.lower()
        if "outbound cold call" in pl:
            return "Hi, this is Ava following up on your recent interest. Did I catch you at an okay time?"
        if "qualifying the prospect" in pl:
            return (
                "Thanks. To make sure this is relevant, are you the person who owns sales process decisions "
                "and are you looking to improve follow-up speed this quarter?"
            )
        if "you are the sdr for an ai outbound" in pl:
            return (
                "We help teams automate outbound follow-up so prospects get contacted quickly and "
                "reps spend more time in qualified conversations. Would you be open to a short walkthrough?"
            )
        if "handling an objection" in pl or "objection handling" in pl:
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
        trace: dict[str, Any] | None = None,
    ) -> str:
        text = human_input.lower().strip()
        labels_set = set(labels)
        # Qualify router: move to pitch once the prospect has given substantive role/engagement
        # (tests rely on these phrases; real LLM routing is smarter).
        if (
            "continue_qualifying" in labels_set
            and "pitch" in labels_set
            and "not_interested" in labels_set
        ):
            if any(
                s in text
                for s in (
                    "oversee",
                    "sales operations",
                    "sales ops",
                    "i lead",
                    "i run sales",
                    "let's do it",
                    "sounds interesting",
                )
            ):
                return "pitch"
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
        trace: dict[str, Any] | None = None,
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

    _RESPOND_TRANSCRIPT_LIMIT = 8
    _EXTRACTION_TRANSCRIPT_LIMIT = 6

    def __init__(self, settings: SDRSettings | None = None, *, bot_config: dict | None = None):
        if bot_config:
            provider = bot_config.get("llm_provider", "openai")
            model_name = bot_config.get("llm_model_name", "gpt-4o-mini")
            temperature = bot_config.get("llm_temperature", 0.4)
            max_tokens = bot_config.get("llm_max_tokens", 220)
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

        self._provider = provider
        self._model_name = model_name
        self._max_tokens = max_tokens

    async def _ainvoke_with_logging(
        self,
        model: Any,
        messages: list[Any],
        *,
        operation: str,
        trace: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        prompt_chars: int | None = None,
        transcript: list[dict[str, str]] | None = None,
        input_text: str | None = None,
        labels: Sequence[str] | None = None,
    ) -> Any:
        call_id = uuid.uuid4().hex[:8]
        preview_source = input_text if input_text is not None else (
            _last_human_text(transcript) if transcript is not None else ""
        )
        transcript_messages = len(transcript) if transcript is not None else "-"
        transcript_human_messages = (
            _count_role_messages(transcript, "human") if transcript is not None else "-"
        )
        transcript_agent_messages = (
            _count_role_messages(transcript, "agent") if transcript is not None else "-"
        )
        logger.info(
            "LLM call start call_id={} conversation_id={} turn_id={} turn_count={} "
            "node={} step={} operation={} provider={} model={} max_tokens={} "
            "message_count={} prompt_chars={} transcript_messages={} "
            "transcript_human_messages={} transcript_agent_messages={} "
            "labels={} input_preview={!r}",
            call_id,
            _trace_value(trace, "conversation_id"),
            _trace_value(trace, "turn_id"),
            _trace_value(trace, "turn_count"),
            _trace_value(trace, "node"),
            _trace_value(trace, "step"),
            operation,
            self._provider,
            self._model_name,
            max_tokens if max_tokens is not None else "-",
            len(messages),
            prompt_chars if prompt_chars is not None else "-",
            transcript_messages,
            transcript_human_messages,
            transcript_agent_messages,
            ",".join(labels) if labels else "-",
            _preview_text(preview_source),
        )
        started_at = time.perf_counter()
        try:
            response = await model.ainvoke(messages)
        except asyncio.CancelledError:
            latency_ms = (time.perf_counter() - started_at) * 1000
            logger.warning(
                "LLM call cancelled call_id={} conversation_id={} turn_id={} turn_count={} "
                "node={} step={} operation={} latency_ms={:.0f}",
                call_id,
                _trace_value(trace, "conversation_id"),
                _trace_value(trace, "turn_id"),
                _trace_value(trace, "turn_count"),
                _trace_value(trace, "node"),
                _trace_value(trace, "step"),
                operation,
                latency_ms,
            )
            raise
        except Exception:
            latency_ms = (time.perf_counter() - started_at) * 1000
            logger.exception(
                "LLM call failed call_id={} conversation_id={} turn_id={} turn_count={} "
                "node={} step={} operation={} latency_ms={:.0f}",
                call_id,
                _trace_value(trace, "conversation_id"),
                _trace_value(trace, "turn_id"),
                _trace_value(trace, "turn_count"),
                _trace_value(trace, "node"),
                _trace_value(trace, "step"),
                operation,
                latency_ms,
            )
            raise
        latency_ms = (time.perf_counter() - started_at) * 1000
        output_text = str(getattr(response, "content", response))
        logger.info(
            "LLM call end call_id={} conversation_id={} turn_id={} turn_count={} "
            "node={} step={} operation={} latency_ms={:.0f} output_chars={} "
            "output_preview={!r}",
            call_id,
            _trace_value(trace, "conversation_id"),
            _trace_value(trace, "turn_id"),
            _trace_value(trace, "turn_count"),
            _trace_value(trace, "node"),
            _trace_value(trace, "step"),
            operation,
            latency_ms,
            len(output_text),
            _preview_text(output_text),
        )
        return response

    async def respond(
        self,
        *,
        system_prompt: str,
        transcript: list[dict[str, str]],
        max_tokens: int | None = None,
        trace: dict[str, Any] | None = None,
    ) -> str:
        trimmed_transcript = _slice_transcript(
            transcript,
            max_messages=self._RESPOND_TRANSCRIPT_LIMIT,
        )
        messages = [SystemMessage(content=system_prompt), *_messages_from_transcript(trimmed_transcript)]
        limit = self._max_tokens if max_tokens is None else min(max_tokens, self._max_tokens)
        model = self._model.bind(max_tokens=limit)
        response = await self._ainvoke_with_logging(
            model,
            messages,
            operation="respond",
            trace=trace,
            max_tokens=limit,
            prompt_chars=len(system_prompt),
            transcript=trimmed_transcript,
        )
        return str(response.content)

    async def classify(
        self,
        *,
        instruction: str,
        human_input: str,
        labels: Sequence[str],
        trace: dict[str, Any] | None = None,
    ) -> str:
        messages = [
            SystemMessage(
                content=(
                    f"{instruction}\n\nReturn exactly one of these labels: "
                    + ", ".join(labels)
                    + "\n\nRespond with ONLY the label, nothing else."
                )
            ),
            HumanMessage(content=human_input),
        ]
        response = await self._ainvoke_with_logging(
            self._model,
            messages,
            operation="classify",
            trace=trace,
            prompt_chars=len(instruction),
            input_text=human_input,
            labels=labels,
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
        trace: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        from ai_sdr_agent.graph.prompts import qualification_extraction_prompt

        system = qualification_extraction_prompt(existing_pain_points)
        trimmed_transcript = _slice_transcript(
            transcript,
            max_messages=self._EXTRACTION_TRANSCRIPT_LIMIT,
        )
        messages = [SystemMessage(content=system), *_messages_from_transcript(trimmed_transcript)]

        response = await self._ainvoke_with_logging(
            self._model,
            messages,
            operation="extract_qualification",
            trace=trace,
            prompt_chars=len(system),
            transcript=trimmed_transcript,
        )
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
