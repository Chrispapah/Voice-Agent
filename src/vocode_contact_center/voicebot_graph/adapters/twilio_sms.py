from __future__ import annotations

import asyncio

from twilio.rest import Client

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

    async def send(self, request: SmsRequest) -> SmsResult:
        try:
            message = await asyncio.to_thread(self._create_message, request)
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

    def _create_message(self, request: SmsRequest):
        payload = {
            "to": request.recipient_phone_number,
            "body": request.message,
        }
        if self._messaging_service_sid:
            payload["messaging_service_sid"] = self._messaging_service_sid
        else:
            payload["from_"] = self._from_number
        return self._client.messages.create(**payload)
