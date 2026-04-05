import asyncio

from vocode_contact_center.settings import ContactCenterSettings
from vocode_contact_center.voicebot_graph.adapters.base import SmsRequest
from vocode_contact_center.voicebot_graph.adapters.twilio_sms import TwilioSmsSender


class _FakeMessage:
    sid = "SMWHATSAPP123"
    status = "queued"


class _FakeMessagesApi:
    def __init__(self):
        self.payloads = []

    def create(self, **payload):
        self.payloads.append(payload)
        return _FakeMessage()


class _FakeTwilioClient:
    def __init__(self, *args, **kwargs):
        self.messages = _FakeMessagesApi()


def test_twilio_sender_formats_whatsapp_addresses(monkeypatch):
    monkeypatch.setattr(
        "vocode_contact_center.voicebot_graph.adapters.twilio_sms.Client",
        _FakeTwilioClient,
    )
    sender = TwilioSmsSender(
        ContactCenterSettings(
            twilio_account_sid="sid",
            twilio_auth_token="token",
            sms_adapter_mode="twilio",
            twilio_message_channel="whatsapp",
            twilio_whatsapp_from_number="+14155238886",
            sms_default_region="US",
        )
    )

    result = asyncio.run(
        sender.send(
            SmsRequest(
                session_id="session-1",
                recipient_phone_number="(415) 555-2671",
                message="Hello from WhatsApp",
                context="registration_confirmation",
            )
        )
    )

    payload = sender._client.messages.payloads[0]
    assert payload["to"] == "whatsapp:+14155552671"
    assert payload["from_"] == "whatsapp:+14155238886"
    assert result.status == "sent"
    assert result.metadata["channel"] == "whatsapp"


def test_missing_sms_values_accepts_whatsapp_sender():
    settings = ContactCenterSettings(
        sms_adapter_mode="twilio",
        twilio_message_channel="whatsapp",
        twilio_account_sid="sid",
        twilio_auth_token="token",
        twilio_whatsapp_from_number="+14155238886",
    )

    assert settings.missing_sms_values() == []
