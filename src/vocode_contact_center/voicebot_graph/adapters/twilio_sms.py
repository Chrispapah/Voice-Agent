from __future__ import annotations

import asyncio

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
        self._client = Client(settings.twilio_account_sid, settings.twilio_auth_token)
        self._from_number = settings.twilio_sms_from_number
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

        metadata = {"provider": "twilio"}
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
        payload = {
            "to": normalized_phone_number,
            "body": request.message,
        }
        if self._messaging_service_sid:
            payload["messaging_service_sid"] = self._messaging_service_sid
        else:
            payload["from_"] = self._from_number
        return self._client.messages.create(**payload)
