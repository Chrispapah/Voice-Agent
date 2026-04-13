from __future__ import annotations

from pydantic import BaseModel, Field


class LeadRecord(BaseModel):
    lead_id: str
    lead_name: str
    company: str
    phone_number: str
    lead_email: str
    lead_context: str = ""
    lifecycle_stage: str = "follow_up"
    timezone: str = "UTC"
    owner_name: str = "Sales Team"
    calendar_id: str = "sales-team"
    metadata: dict[str, str] = Field(default_factory=dict)
