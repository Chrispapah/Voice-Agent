from __future__ import annotations

from vocode_contact_center.phone_numbers import (
    extract_caller_e164_from_call_context,
    normalize_phone_number_cached,
)
from vocode_contact_center.settings import ContactCenterSettings
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
    def __init__(self, settings: ContactCenterSettings | None = None) -> None:
        self._settings = settings or ContactCenterSettings()

    async def authenticate(self, request: AuthenticationRequest) -> AuthenticationResult:
        data = dict(request.collected_data)
        context = request.interaction_context or "login"

        if context == "registration":
            if "full_name" not in data:
                return AuthenticationResult(
                    status="needs_customer_input",
                    prompt=(
                        "I need your first and last name for this registration. "
                        "Please say both names clearly."
                    ),
                    requested_field="full_name",
                )
            if "phone_number" not in data and self._settings.sms_prefer_caller_id_for_registration:
                ani = extract_caller_e164_from_call_context(request.call_context)
                if ani:
                    data["phone_number"] = ani
            if "phone_number" not in data:
                return AuthenticationResult(
                    status="needs_customer_input",
                    prompt=(
                        "Please share the mobile number we should use for registration. "
                        "Say plus, then your country code digit by digit—for example plus three zero for Greece—"
                        "then your number digit by digit. You can also say country code, plus three zero, "
                        "phone number, then the rest of your digits."
                    ),
                    requested_field="phone_number",
                )
            normalized_phone_number = normalize_phone_number_cached(
                data.get("phone_number", ""),
                self._settings.sms_default_region,
            )
            if not normalized_phone_number and self._settings.sms_prefer_caller_id_for_registration:
                ani = extract_caller_e164_from_call_context(request.call_context)
                if ani:
                    normalized_phone_number = normalize_phone_number_cached(
                        ani, self._settings.sms_default_region
                    )
            if not normalized_phone_number:
                return AuthenticationResult(
                    status="needs_customer_input",
                    prompt=(
                        "That phone number did not sound valid. Please say it again slowly, "
                        "with plus and your country code digit by digit, then your mobile number digit by digit."
                    ),
                    requested_field="phone_number",
                )
            data["phone_number"] = normalized_phone_number
            if data.get("phone_number", "").endswith("0000"):
                return AuthenticationResult(
                    status="failure",
                    prompt="I could not validate that registration request with the current details.",
                    metadata={"reason": "stub_registration_denied"},
                )
            if data.get("sms_confirmed") != "true":
                return AuthenticationResult(
                    status="needs_sms_confirmation",
                    prompt="I can now send an SMS confirmation to finish registration.",
                    normalized_data=data,
                )
            expected_code = "".join(ch for ch in data.get("expected_verification_code", "") if ch.isdigit())
            provided_code = "".join(ch for ch in data.get("confirmation_code", "") if ch.isdigit())
            if "confirmation_code" not in data:
                return AuthenticationResult(
                    status="needs_customer_input",
                    prompt="Please tell me the confirmation code that was sent to your phone.",
                    requested_field="confirmation_code",
                    normalized_data=data,
                )
            if not expected_code or provided_code != expected_code:
                data.pop("confirmation_code", None)
                data["sms_confirmed"] = "false"
                return AuthenticationResult(
                    status="needs_customer_input",
                    prompt="That code doesn't match the one I sent. Please tell me the confirmation code again.",
                    requested_field="confirmation_code",
                    normalized_data=data,
                    metadata={"reason": "verification_code_mismatch"},
                )
            if "full_id_number" not in data:
                return AuthenticationResult(
                    status="needs_customer_input",
                    prompt="Thanks. Now please tell me your full ID number.",
                    requested_field="full_id_number",
                    normalized_data=data,
                )
            return AuthenticationResult(
                status="success",
                prompt="Registration verification is complete.",
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
