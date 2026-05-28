from __future__ import annotations

import asyncio
import json
import time
import uuid
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Awaitable, Callable, Protocol, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from loguru import logger

try:
    from langchain_groq import ChatGroq
except ImportError:  # pragma: no cover
    ChatGroq = None

try:
    from groq import AsyncGroq
except ImportError:  # pragma: no cover
    AsyncGroq = None

from ai_sdr_agent.config import SDRSettings, get_settings

_GROQ_FALLBACK_MODEL = "llama-3.3-70b-versatile"


def _is_likely_non_groq_chat_model(model_name: str) -> bool:
    u = (model_name or "").strip().lower()
    if not u:
        return True
    return (
        u.startswith("gpt-")
        or u.startswith("claude")
        or u.startswith("o1")
        or u.startswith("o3")
        or "davinci" in u
    )

ResponseChunkSink = Callable[[str], Awaitable[None]]
_RESPONSE_CHUNK_SINK: ContextVar[ResponseChunkSink | None] = ContextVar(
    "ai_sdr_response_chunk_sink",
    default=None,
)


def set_response_chunk_sink(sink: ResponseChunkSink) -> Token[ResponseChunkSink | None]:
    return _RESPONSE_CHUNK_SINK.set(sink)


def reset_response_chunk_sink(token: Token[ResponseChunkSink | None]) -> None:
    _RESPONSE_CHUNK_SINK.reset(token)


def get_response_chunk_sink() -> ResponseChunkSink | None:
    return _RESPONSE_CHUNK_SINK.get()


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
) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    for item in transcript:
        if item["role"] == "human":
            messages.append(HumanMessage(content=item["content"]))
        else:
            messages.append(AIMessage(content=item["content"]))
    return messages


def _groq_messages_from_transcript(
    *,
    system_prompt: str,
    transcript: list[dict[str, str]],
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for item in transcript:
        if item["role"] == "human":
            messages.append({"role": "user", "content": item["content"]})
        else:
            messages.append({"role": "assistant", "content": item["content"]})
    return messages


@dataclass(frozen=True)
class ToolDefinition:
    """LLM-facing tool description (OpenAI/Groq function-calling schema).

    ``parameters`` is a JSON Schema object describing the tool arguments.
    """

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# Async callable: (tool_name, arguments_dict) -> tool_result_string.
ToolExecutor = Callable[[str, dict[str, Any]], Awaitable[str]]


class ConversationBrain(Protocol):
    def supports_response_token_stream(self) -> bool:
        ...

    async def respond(
        self,
        *,
        system_prompt: str,
        transcript: list[dict[str, str]],
        max_tokens: int | None = None,
        trace: dict[str, Any] | None = None,
    ) -> str:
        ...

    async def stream_respond_tokens(
        self,
        *,
        system_prompt: str,
        transcript: list[dict[str, str]],
        max_tokens: int | None = None,
        trace: dict[str, Any] | None = None,
    ) -> AsyncGenerator[str, None]:
        ...

    async def respond_with_tools(
        self,
        *,
        system_prompt: str,
        transcript: list[dict[str, str]],
        tools: Sequence[ToolDefinition],
        tool_executor: ToolExecutor,
        max_tokens: int | None = None,
        trace: dict[str, Any] | None = None,
        max_tool_iterations: int = 2,
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


@dataclass
class StubConversationBrain:
    def supports_response_token_stream(self) -> bool:
        return False

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
        _ = self._last_human_text(transcript)
        return "Καταλαβαίνω. Πώς μπορώ να σας βοηθήσω;"

    async def stream_respond_tokens(
        self,
        *,
        system_prompt: str,
        transcript: list[dict[str, str]],
        max_tokens: int | None = None,
        trace: dict[str, Any] | None = None,
    ) -> AsyncGenerator[str, None]:
        response = await self.respond(
            system_prompt=system_prompt,
            transcript=transcript,
            max_tokens=max_tokens,
            trace=trace,
        )
        if response:
            yield response

    async def respond_with_tools(
        self,
        *,
        system_prompt: str,
        transcript: list[dict[str, str]],
        tools: Sequence[ToolDefinition],
        tool_executor: ToolExecutor,
        max_tokens: int | None = None,
        trace: dict[str, Any] | None = None,
        max_tool_iterations: int = 2,
    ) -> str:
        # The stub brain has no real model, so it cannot decide to invoke tools.
        # Tests and offline runs get deterministic behaviour by delegating to respond().
        return await self.respond(
            system_prompt=system_prompt,
            transcript=transcript,
            max_tokens=max_tokens,
            trace=trace,
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
        if any(phrase in text for phrase in self._EXIT_PHRASES):
            if "complete" in labels:
                return "complete"
        if text in self._NEGATIVE_PHRASES or any(text.startswith(p) for p in self._NEGATIVE_PHRASES):
            if "complete" in labels:
                return "complete"
        return labels[0]


class LangChainConversationBrain:
    """Conversation LLM (respond, classify, extract) always uses Groq via bot or env API key."""

    _RESPOND_TRANSCRIPT_LIMIT = 8
    _EXTRACTION_TRANSCRIPT_LIMIT = 6

    def __init__(self, settings: SDRSettings | None = None, *, bot_config: dict | None = None):
        env = settings if settings is not None else get_settings()
        if bot_config:
            model_name = bot_config.get("llm_model_name", _GROQ_FALLBACK_MODEL)
            temperature = bot_config.get("llm_temperature", 0.4)
            max_tokens = bot_config.get("llm_max_tokens", 220)
            groq_key = bot_config.get("groq_api_key") or env.groq_api_key
        elif settings:
            model_name = settings.llm_model_name
            temperature = settings.llm_temperature
            max_tokens = settings.llm_max_tokens
            groq_key = settings.groq_api_key
        else:
            raise ValueError("Either settings or bot_config must be provided")

        if _is_likely_non_groq_chat_model(model_name):
            candidate = env.llm_model_name
            model_name = candidate if not _is_likely_non_groq_chat_model(candidate) else _GROQ_FALLBACK_MODEL

        if not groq_key:
            raise ValueError("GROQ_API_KEY or bot groq_api_key is required (all conversation LLM uses Groq).")
        if ChatGroq is None:
            raise ValueError("langchain_groq is required for ConversationBrain.")

        self._model = ChatGroq(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=groq_key,
        )
        self._provider = "groq"
        self._model_name = model_name
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._groq_async_client = (
            AsyncGroq(api_key=groq_key) if AsyncGroq is not None and groq_key else None
        )

    def supports_response_token_stream(self) -> bool:
        return self._provider == "groq" and self._groq_async_client is not None

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
        sink = get_response_chunk_sink()
        if sink is not None and self.supports_response_token_stream():
            parts: list[str] = []
            async for chunk in self.stream_respond_tokens(
                system_prompt=system_prompt,
                transcript=transcript,
                max_tokens=max_tokens,
                trace=trace,
            ):
                if not chunk:
                    continue
                parts.append(chunk)
                await sink(chunk)
            return "".join(parts)

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

    async def stream_respond_tokens(
        self,
        *,
        system_prompt: str,
        transcript: list[dict[str, str]],
        max_tokens: int | None = None,
        trace: dict[str, Any] | None = None,
    ) -> AsyncGenerator[str, None]:
        if not self.supports_response_token_stream():
            response = await self.respond(
                system_prompt=system_prompt,
                transcript=transcript,
                max_tokens=max_tokens,
                trace=trace,
            )
            if response:
                yield response
            return

        trimmed_transcript = _slice_transcript(
            transcript,
            max_messages=self._RESPOND_TRANSCRIPT_LIMIT,
        )
        limit = self._max_tokens if max_tokens is None else min(max_tokens, self._max_tokens)
        messages = _groq_messages_from_transcript(
            system_prompt=system_prompt,
            transcript=trimmed_transcript,
        )
        call_id = uuid.uuid4().hex[:8]
        logger.info(
            "LLM stream start call_id={} conversation_id={} turn_id={} turn_count={} "
            "node={} step={} operation={} provider={} model={} max_tokens={} "
            "message_count={} prompt_chars={} transcript_messages={} "
            "transcript_human_messages={} transcript_agent_messages={} "
            "input_preview={!r}",
            call_id,
            _trace_value(trace, "conversation_id"),
            _trace_value(trace, "turn_id"),
            _trace_value(trace, "turn_count"),
            _trace_value(trace, "node"),
            _trace_value(trace, "step"),
            "respond_stream",
            self._provider,
            self._model_name,
            limit,
            len(messages),
            len(system_prompt),
            len(trimmed_transcript),
            _count_role_messages(trimmed_transcript, "human"),
            _count_role_messages(trimmed_transcript, "agent"),
            _preview_text(_last_human_text(trimmed_transcript)),
        )
        started_at = time.perf_counter()
        output_parts: list[str] = []
        emitted_first_token = False
        stream = None
        try:
            assert self._groq_async_client is not None
            stream = await self._groq_async_client.chat.completions.create(
                messages=messages,
                model=self._model_name,
                temperature=self._temperature,
                max_tokens=limit,
                stream=True,
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if not delta:
                    continue
                if not emitted_first_token:
                    emitted_first_token = True
                    logger.info(
                        "LLM stream first_token call_id={} conversation_id={} turn_id={} turn_count={} "
                        "node={} step={} operation={} latency_ms={:.0f}",
                        call_id,
                        _trace_value(trace, "conversation_id"),
                        _trace_value(trace, "turn_id"),
                        _trace_value(trace, "turn_count"),
                        _trace_value(trace, "node"),
                        _trace_value(trace, "step"),
                        "respond_stream",
                        (time.perf_counter() - started_at) * 1000,
                    )
                output_parts.append(delta)
                yield delta
        except asyncio.CancelledError:
            latency_ms = (time.perf_counter() - started_at) * 1000
            logger.warning(
                "LLM stream cancelled call_id={} conversation_id={} turn_id={} turn_count={} "
                "node={} step={} operation={} latency_ms={:.0f}",
                call_id,
                _trace_value(trace, "conversation_id"),
                _trace_value(trace, "turn_id"),
                _trace_value(trace, "turn_count"),
                _trace_value(trace, "node"),
                _trace_value(trace, "step"),
                "respond_stream",
                latency_ms,
            )
            raise
        except Exception:
            latency_ms = (time.perf_counter() - started_at) * 1000
            logger.exception(
                "LLM stream failed call_id={} conversation_id={} turn_id={} turn_count={} "
                "node={} step={} operation={} latency_ms={:.0f}",
                call_id,
                _trace_value(trace, "conversation_id"),
                _trace_value(trace, "turn_id"),
                _trace_value(trace, "turn_count"),
                _trace_value(trace, "node"),
                _trace_value(trace, "step"),
                "respond_stream",
                latency_ms,
            )
            raise
        finally:
            aclose = getattr(stream, "aclose", None)
            if aclose is not None:
                await aclose()
        output_text = "".join(output_parts)
        logger.info(
            "LLM stream end call_id={} conversation_id={} turn_id={} turn_count={} "
            "node={} step={} operation={} latency_ms={:.0f} output_chars={} "
            "output_preview={!r}",
            call_id,
            _trace_value(trace, "conversation_id"),
            _trace_value(trace, "turn_id"),
            _trace_value(trace, "turn_count"),
            _trace_value(trace, "node"),
            _trace_value(trace, "step"),
            "respond_stream",
            (time.perf_counter() - started_at) * 1000,
            len(output_text),
            _preview_text(output_text),
        )

    async def respond_with_tools(
        self,
        *,
        system_prompt: str,
        transcript: list[dict[str, str]],
        tools: Sequence[ToolDefinition],
        tool_executor: ToolExecutor,
        max_tokens: int | None = None,
        trace: dict[str, Any] | None = None,
        max_tool_iterations: int = 2,
    ) -> str:
        """Single-turn answer that may invoke ``tools`` zero or more times.

        Streams text deltas to the active chunk sink (so TTS streaming still works
        when the model is just answering). Tool calls are buffered, executed via
        ``tool_executor``, and the loop continues. After ``max_tool_iterations``
        we force a final tool-less call to guarantee the model produces speech.
        """
        if not tools or self._groq_async_client is None:
            return await self.respond(
                system_prompt=system_prompt,
                transcript=transcript,
                max_tokens=max_tokens,
                trace=trace,
            )

        trimmed = _slice_transcript(transcript, max_messages=self._RESPOND_TRANSCRIPT_LIMIT)
        limit = self._max_tokens if max_tokens is None else min(max_tokens, self._max_tokens)
        messages = _groq_messages_from_transcript(
            system_prompt=system_prompt,
            transcript=trimmed,
        )
        tool_schema = [t.to_openai_schema() for t in tools]
        sink = get_response_chunk_sink()
        call_id = uuid.uuid4().hex[:8]
        started_at = time.perf_counter()
        logger.info(
            "LLM tools start call_id={} conversation_id={} turn_id={} turn_count={} "
            "node={} step={} provider={} model={} tool_count={} max_iterations={} "
            "max_tokens={} message_count={}",
            call_id,
            _trace_value(trace, "conversation_id"),
            _trace_value(trace, "turn_id"),
            _trace_value(trace, "turn_count"),
            _trace_value(trace, "node"),
            _trace_value(trace, "step"),
            self._provider,
            self._model_name,
            len(tools),
            max_tool_iterations,
            limit,
            len(messages),
        )

        executed_tool_calls = 0
        last_content = ""

        for iteration in range(max_tool_iterations + 1):
            is_last_iteration = iteration == max_tool_iterations
            stream = None
            content_parts: list[str] = []
            tool_calls_acc: dict[int, dict[str, str]] = {}
            finish_reason: str | None = None
            iter_started_at = time.perf_counter()

            try:
                create_kwargs: dict[str, Any] = {
                    "messages": messages,
                    "model": self._model_name,
                    "temperature": self._temperature,
                    "max_tokens": limit,
                    "stream": True,
                }
                if not is_last_iteration:
                    create_kwargs["tools"] = tool_schema
                    create_kwargs["tool_choice"] = "auto"

                stream = await self._groq_async_client.chat.completions.create(**create_kwargs)
                async for chunk in stream:
                    if not chunk.choices:
                        continue
                    choice = chunk.choices[0]
                    delta = choice.delta
                    if delta is not None:
                        content_delta = getattr(delta, "content", None)
                        if content_delta:
                            content_parts.append(content_delta)
                            if sink is not None:
                                await sink(content_delta)
                        tool_call_deltas = getattr(delta, "tool_calls", None) or []
                        for tc_delta in tool_call_deltas:
                            idx = getattr(tc_delta, "index", 0) or 0
                            bucket = tool_calls_acc.setdefault(
                                idx, {"id": "", "name": "", "args": ""}
                            )
                            tc_id = getattr(tc_delta, "id", None)
                            if tc_id:
                                bucket["id"] = tc_id
                            fn = getattr(tc_delta, "function", None)
                            if fn is not None:
                                fn_name = getattr(fn, "name", None)
                                if fn_name:
                                    bucket["name"] = fn_name
                                fn_args = getattr(fn, "arguments", None)
                                if fn_args:
                                    bucket["args"] += fn_args
                    if choice.finish_reason:
                        finish_reason = choice.finish_reason
            except asyncio.CancelledError:
                logger.warning(
                    "LLM tools cancelled call_id={} iteration={} latency_ms={:.0f}",
                    call_id,
                    iteration,
                    (time.perf_counter() - iter_started_at) * 1000,
                )
                raise
            except Exception:
                logger.exception(
                    "LLM tools iteration failed call_id={} iteration={}",
                    call_id,
                    iteration,
                )
                raise
            finally:
                aclose = getattr(stream, "aclose", None)
                if aclose is not None:
                    try:
                        await aclose()
                    except Exception:
                        pass

            last_content = "".join(content_parts)

            if not tool_calls_acc:
                logger.info(
                    "LLM tools done call_id={} iterations={} tool_calls_executed={} "
                    "finish_reason={} output_chars={} total_latency_ms={:.0f}",
                    call_id,
                    iteration + 1,
                    executed_tool_calls,
                    finish_reason or "-",
                    len(last_content),
                    (time.perf_counter() - started_at) * 1000,
                )
                return last_content

            tool_calls_payload = [
                {
                    "id": tc["id"] or f"call_{call_id}_{idx}",
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": tc["args"] or "{}",
                    },
                }
                for idx, tc in sorted(tool_calls_acc.items())
            ]
            messages.append(
                {
                    "role": "assistant",
                    "content": last_content or None,
                    "tool_calls": tool_calls_payload,
                }
            )

            for tc_payload in tool_calls_payload:
                name = tc_payload["function"]["name"]
                raw_args = tc_payload["function"]["arguments"]
                try:
                    args = json.loads(raw_args) if raw_args else {}
                except json.JSONDecodeError:
                    args = {}
                tool_started_at = time.perf_counter()
                try:
                    result = await tool_executor(name, args)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.exception(
                        "Tool executor failed call_id={} tool={} args={!r}",
                        call_id,
                        name,
                        raw_args,
                    )
                    result = f"[tool error: {exc.__class__.__name__}]"
                executed_tool_calls += 1
                logger.info(
                    "LLM tool exec call_id={} iteration={} tool={} args_chars={} "
                    "result_chars={} latency_ms={:.0f}",
                    call_id,
                    iteration,
                    name,
                    len(raw_args or ""),
                    len(result or ""),
                    (time.perf_counter() - tool_started_at) * 1000,
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc_payload["id"],
                        "name": name,
                        "content": result or "",
                    }
                )

        logger.warning(
            "LLM tools exhausted iterations call_id={} executed={} latency_ms={:.0f}",
            call_id,
            executed_tool_calls,
            (time.perf_counter() - started_at) * 1000,
        )
        return last_content

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


def build_conversation_brain(
    settings: SDRSettings | None = None,
    *,
    bot_config: dict | None = None,
) -> ConversationBrain:
    """Build a brain from SDRSettings (legacy) or a per-bot config dict."""
    env_settings = settings if settings is not None else get_settings()
    provider = (
        (bot_config or {}).get("llm_provider")
        or env_settings.llm_provider
    )
    if provider == "stub":
        return StubConversationBrain()
    return LangChainConversationBrain(env_settings, bot_config=bot_config)
