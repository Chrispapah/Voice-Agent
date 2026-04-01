from __future__ import annotations

from vocode_contact_center.voicebot_graph.adapters.base import (
    AuthenticationAdapter,
    AuthenticationRequest,
    AuthenticationResult,
    GenesysAdapter,
    GenesysRequest,
    GenesysResult,
    SmsRequest,
    SmsResult,
    SmsSender,
)


class StubAuthenticationAdapter(AuthenticationAdapter):
    async def authenticate(self, request: AuthenticationRequest) -> AuthenticationResult:
        data = dict(request.collected_data)
        context = request.interaction_context or "login"

        if context == "registration":
            if "full_name" not in data:
                return AuthenticationResult(
                    status="needs_customer_input",
                    prompt="To continue with registration, please tell me your full name.",
                    requested_field="full_name",
                )
            if "phone_number" not in data:
                return AuthenticationResult(
                    status="needs_customer_input",
                    prompt="Please share the phone number we should use for registration.",
                    requested_field="phone_number",
                )
            if data.get("phone_number", "").endswith("0000"):
                return AuthenticationResult(
                    status="failure",
                    prompt="I could not validate that registration request with the current details.",
                    metadata={"reason": "stub_registration_denied"},
                )
            return AuthenticationResult(
                status="needs_sms_confirmation",
                prompt="I can now send an SMS confirmation to finish registration.",
                normalized_data=data,
            )

        if "account_id" not in data:
            return AuthenticationResult(
                status="needs_customer_input",
                prompt="Please tell me your account ID or phone number so I can verify your login.",
                requested_field="account_id",
            )
        if "password_hint" not in data:
            return AuthenticationResult(
                status="needs_customer_input",
                prompt="I need one more detail. Please tell me the password hint or last known code on the account.",
                requested_field="password_hint",
            )
        if any(flag in data.get("account_id", "").lower() for flag in ("fail", "wrong", "invalid")):
            return AuthenticationResult(
                status="failure",
                prompt="The authentication system could not verify that account.",
                metadata={"reason": "stub_login_denied"},
            )
        return AuthenticationResult(
            status="success",
            prompt="Authentication succeeded.",
            normalized_data=data,
        )


class StubGenesysAdapter(GenesysAdapter):
    async def connect(self, request: GenesysRequest) -> GenesysResult:
        queue_name = "announcements_queue" if request.path_name == "announcements" else "feedback_queue"
        return GenesysResult(
            status="connected",
            prompt="I connected the request to the contact center routing system.",
            metadata={
                "queue": queue_name,
                "path_name": request.path_name,
            },
        )


class StubSmsSender(SmsSender):
    async def send(self, request: SmsRequest) -> SmsResult:
        return SmsResult(
            status="sent",
            provider_message_id=f"stub-{request.session_id}",
            metadata={
                "provider": "stub",
                "recipient_phone_number": request.recipient_phone_number,
                "context": request.context,
            },
        )
