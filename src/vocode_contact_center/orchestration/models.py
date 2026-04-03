from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncGenerator, Protocol

from pydantic import BaseModel, Field


class ConversationStage(str, Enum):
    ROOT = "root"
    INFORMATION = "information"
    AUTHENTICATION = "authentication"
    REGISTRATION_TERMINAL = "registration_terminal"
    LOGIN_TERMINAL = "login_terminal"
    FAIL_TERMINAL = "fail_terminal"
    ANNOUNCEMENTS_CONTINUE = "announcements_continue"
    ANNOUNCEMENTS_TERMINAL = "announcements_terminal"
    FEEDBACK_QUESTION = "feedback_question"
    FEEDBACK_TERMINAL = "feedback_terminal"


class ConversationIntent(str, Enum):
    INFORMATION = "information"
    INTERACTION = "interaction"
    ANNOUNCEMENTS = "announcements"
    FEEDBACK = "feedback"
    GENERAL = "general"


class InteractionContext(str, Enum):
    REGISTRATION = "registration"
    LOGIN = "login"


class PolicyAction(str, Enum):
    SELECT_OPTION = "select_option"
    ASK_CLARIFYING_QUESTION = "ask_clarifying_question"
    ANSWER_DIRECTLY = "answer_directly"
    FALLBACK = "fallback"


class ConversationSessionState(BaseModel):
    session_id: str
    call_context: str
    metadata: dict[str, str] = Field(default_factory=dict)
    history: list[tuple[str, str]] = Field(default_factory=list)
    stage: ConversationStage = ConversationStage.ROOT
    current_intent: ConversationIntent | None = None
    interaction_context: InteractionContext | None = None
    pending_auth_field: str | None = None
    auth_attempts: int = 0
    collected_data: dict[str, str] = Field(default_factory=dict)
    available_options: list[str] = Field(default_factory=list)
    latest_user_input: str = ""
    response_text: str = ""
    final_outcome: str | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)
    adapter_results: dict[str, Any] = Field(default_factory=dict)
    last_completed_path: str | None = None
    genesys_requested: bool = False


class ConversationPolicyDecision(BaseModel):
    action: PolicyAction
    selected_option: str | None = None
    response_text: str = Field(
        default="",
        description="The natural-language text the assistant should say next.",
    )


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
