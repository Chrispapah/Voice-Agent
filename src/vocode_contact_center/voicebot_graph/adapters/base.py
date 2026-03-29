from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class AuthenticationRequest:
    session_id: str
    call_context: str
    interaction_context: str | None
    latest_user_input: str
    collected_data: dict[str, str]
    auth_attempts: int


@dataclass
class AuthenticationResult:
    status: str
    prompt: str
    requested_field: str | None = None
    normalized_data: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class GenesysRequest:
    session_id: str
    path_name: str
    latest_user_input: str
    metadata: dict[str, str]


@dataclass
class GenesysResult:
    status: str
    prompt: str
    metadata: dict[str, str] = field(default_factory=dict)


class AuthenticationAdapter(Protocol):
    async def authenticate(self, request: AuthenticationRequest) -> AuthenticationResult:
        ...


class GenesysAdapter(Protocol):
    async def connect(self, request: GenesysRequest) -> GenesysResult:
        ...
