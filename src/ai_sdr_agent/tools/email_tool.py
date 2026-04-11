from __future__ import annotations

from pathlib import Path
from typing import Protocol

from jinja2 import Template
from langchain_core.tools import StructuredTool
from pydantic import BaseModel


class FollowUpEmailRequest(BaseModel):
    to_email: str
    subject: str
    body: str
    from_name: str


class EmailGateway(Protocol):
    async def send_email(
        self,
        *,
        to_email: str,
        subject: str,
        body: str,
        from_name: str,
    ) -> dict[str, str]:
        ...


class StubEmailGateway:
    def __init__(self):
        self.sent_messages: list[dict[str, str]] = []

    async def send_email(
        self,
        *,
        to_email: str,
        subject: str,
        body: str,
        from_name: str,
    ) -> dict[str, str]:
        payload = {
            "to_email": to_email,
            "subject": subject,
            "body": body,
            "from_name": from_name,
            "message_id": f"stub-email-{len(self.sent_messages) + 1}",
        }
        self.sent_messages.append(payload)
        return payload


def render_follow_up_email(
    *,
    template_path: Path,
    lead_name: str,
    company: str,
    sales_rep_name: str,
    meeting_slot: str | None,
    follow_up_summary: str,
) -> str:
    template = Template(template_path.read_text(encoding="utf-8"))
    return template.render(
        lead_name=lead_name,
        company=company,
        sales_rep_name=sales_rep_name,
        meeting_slot=meeting_slot,
        follow_up_summary=follow_up_summary,
    )


def build_email_tools(email_gateway: EmailGateway) -> list[StructuredTool]:
    async def send_follow_up_email(
        to_email: str,
        subject: str,
        body: str,
        from_name: str,
    ) -> dict[str, str]:
        return await email_gateway.send_email(
            to_email=to_email,
            subject=subject,
            body=body,
            from_name=from_name,
        )

    return [
        StructuredTool.from_function(
            coroutine=send_follow_up_email,
            name="send_follow_up_email",
            description="Send a follow-up email after an SDR call.",
            args_schema=FollowUpEmailRequest,
        )
    ]
