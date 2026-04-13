from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncGenerator, Protocol


@dataclass
class ConversationTurnResult:
    text: str
    final_outcome: str | None
    active_menu: str | None
    menu_options: list[str]
    artifacts: dict[str, str]
    adapter_results: dict[str, object]
    state_snapshot: dict[str, Any]


class ConversationOrchestrator(Protocol):
    async def run_turn(
        self,
        session_id: str,
        user_text: str,
        *,
        call_context: str,
        metadata: dict[str, str] | None = None,
        commit: bool = True,
    ) -> ConversationTurnResult:
        ...

    async def preview_turn(
        self,
        session_id: str,
        user_text: str,
        *,
        call_context: str,
        metadata: dict[str, str] | None = None,
    ) -> ConversationTurnResult:
        ...

    async def stream_text_response(self, text: str) -> AsyncGenerator[str, None]:
        ...

    async def stream_generate_response(
        self,
        session_id: str,
        user_text: str,
        *,
        call_context: str,
        metadata: dict[str, str] | None = None,
        commit: bool = True,
    ) -> AsyncGenerator[str, None]:
        ...
