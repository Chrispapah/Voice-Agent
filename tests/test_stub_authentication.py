import asyncio

from vocode_contact_center.settings import ContactCenterSettings
from vocode_contact_center.voicebot_graph.adapters.base import AuthenticationRequest
from vocode_contact_center.voicebot_graph.adapters.stub import StubAuthenticationAdapter


def test_stub_registration_prefills_phone_from_caller_id_when_enabled():
    settings = ContactCenterSettings(
        sms_prefer_caller_id_for_registration=True,
        sms_default_region="GR",
    )
    adapter = StubAuthenticationAdapter(settings)
    ctx = "Live call metadata:\n- Caller number: sip:+306988039971@pbx.example.com\n"
    request = AuthenticationRequest(
        session_id="sid",
        call_context=ctx,
        interaction_context="registration",
        latest_user_input="",
        collected_data={"full_name": "Alex Example"},
        auth_attempts=1,
    )

    result = asyncio.run(adapter.authenticate(request))
    assert result.status == "needs_sms_confirmation"
    assert result.normalized_data.get("phone_number") == "+306988039971"


def test_stub_registration_falls_back_to_caller_id_after_invalid_spoken_number():
    settings = ContactCenterSettings(
        sms_prefer_caller_id_for_registration=True,
        sms_default_region="GR",
    )
    adapter = StubAuthenticationAdapter(settings)
    ctx = "Live call metadata:\n- Caller number: sip:+306988039971@pbx.example.com\n"
    request = AuthenticationRequest(
        session_id="sid",
        call_context=ctx,
        interaction_context="registration",
        latest_user_input="gibberish",
        collected_data={"full_name": "Alex Example", "phone_number": "not a phone"},
        auth_attempts=2,
    )

    result = asyncio.run(adapter.authenticate(request))
    assert result.status == "needs_sms_confirmation"
    assert result.normalized_data.get("phone_number") == "+306988039971"
