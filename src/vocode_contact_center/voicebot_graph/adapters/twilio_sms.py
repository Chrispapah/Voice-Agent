from __future__ import annotations

import asyncio
import json
import secrets

from twilio.rest import Client

from vocode_contact_center.phone_numbers import normalize_phone_number
from vocode_contact_center.settings import ContactCenterSettings
from vocode_contact_center.voicebot_graph.adapters.base import (
    SmsRequest,
    SmsResult,
    SmsSender,
)


class TwilioSmsSender(SmsSender):
    def __init__(self, settings: ContactCenterSettings) -> None:
        self._settings = settings
        self._client = Client(settings.twilio_account_sid, settings.twilio_auth_token)
        self._channel = settings.normalized_twilio_message_channel()
        self._from_number = settings.twilio_outbound_from_number()
        self._messaging_service_sid = settings.twilio_messaging_service_sid
        self._default_region = settings.sms_default_region
        self._timeout = max(1.0, float(settings.twilio_sms_timeout_seconds))

    async def send(self, request: SmsRequest) -> SmsResult:
        normalized_phone_number = normalize_phone_number(
            request.recipient_phone_number,
            default_region=self._default_region,
        )
        if not normalized_phone_number:
            return SmsResult(
                status="failed",
                error_message="The destination phone number was not valid.",
                metadata={"provider": "twilio", "reason": "invalid_destination_number"},
            )

        try:
            message = await asyncio.wait_for(
                asyncio.to_thread(
                    self._create_message,
                    request,
                    normalized_phone_number,
                ),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError:
            return SmsResult(
                status="failed",
                error_message="SMS request timed out.",
                metadata={"provider": "twilio", "reason": "timeout"},
            )
        except Exception as exc:
            return SmsResult(
                status="failed",
                error_message=str(exc),
                metadata={"provider": "twilio"},
            )

        metadata = {"provider": "twilio", "channel": self._channel}
        provider_status = getattr(message, "status", None)
        if provider_status:
            metadata["provider_status"] = str(provider_status)
        if self._messaging_service_sid:
            metadata["messaging_service_sid"] = self._messaging_service_sid
        elif self._from_number:
            metadata["from_number"] = self._from_number

        return SmsResult(
            status="sent",
            provider_message_id=getattr(message, "sid", None),
            metadata=metadata,
        )

    def _create_message(self, request: SmsRequest, normalized_phone_number: str):
        payload = {"to": self._format_recipient(normalized_phone_number)}
        template_sid = self._settings.whatsapp_template_sid_for_context(request.context)
        if template_sid:
            payload["content_sid"] = template_sid
            payload["content_variables"] = json.dumps(
                {"1": self._resolve_verification_code(request)}
            )
        else:
            payload["body"] = request.message
        if self._messaging_service_sid:
            payload["messaging_service_sid"] = self._messaging_service_sid
        else:
            payload["from_"] = self._format_sender(self._from_number)
        return self._client.messages.create(**payload)

    def _resolve_verification_code(self, request: SmsRequest) -> str:
        code = (request.metadata or {}).get("verification_code")
        if code:
            return str(code)
        return f"{secrets.randbelow(1_000_000):06d}"

    def _format_recipient(self, normalized_phone_number: str) -> str:
        if self._channel == "whatsapp":
            return f"whatsapp:{normalized_phone_number}"
        return normalized_phone_number

    def _format_sender(self, from_number: str | None) -> str | None:
        if from_number is None:
            return None
        if self._channel == "whatsapp" and not from_number.startswith("whatsapp:"):
            return f"whatsapp:{from_number}"
        return from_number
