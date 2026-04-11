from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


CallOutcome = Literal[
    "meeting_booked",
    "follow_up_needed",
    "not_interested",
    "no_answer",
    "voicemail",
]


class CallLogRecord(BaseModel):
    conversation_id: str
    lead_id: str
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
    call_outcome: CallOutcome = "follow_up_needed"
    transcript: list[dict[str, str]] = Field(default_factory=list)
    qualification_notes: dict[str, str | bool | None] = Field(default_factory=dict)
    meeting_booked: bool = False
    proposed_slot: str | None = None
    follow_up_action: str | None = None
