import asyncio

from vocode_contact_center.orchestration import (
    ConversationPolicyDecision,
    LLMConversationOrchestratorService,
    PolicyAction,
)
from vocode_contact_center.settings import ContactCenterSettings
from vocode_contact_center.voicebot_graph.adapters.base import SmsRequest, SmsResult


def make_settings() -> ContactCenterSettings:
    return ContactCenterSettings(
        conversation_orchestrator="llm_orchestrator",
        langchain_provider="openai",
        openai_api_key="openai",
        sms_default_region="US",
        information_store_website_url="https://demo.example.com/store",
        information_products_pdf_url="https://demo.example.com/products.pdf",
        announcements_message="These are today's announcements.",
        feedback_question_prompt="Would you like to go back to chat or speak to support?",
    )


class QueuePolicy:
    def __init__(self, decisions: list[ConversationPolicyDecision]):
        self._decisions = list(decisions)

    async def decide(self, **kwargs) -> ConversationPolicyDecision:
        if not self._decisions:
            raise AssertionError("No queued policy decision remained for this turn.")
        return self._decisions.pop(0)


class FakeSmsSender:
    def __init__(self, *, status: str = "sent"):
        self.status = status
        self.requests: list[SmsRequest] = []

    async def send(self, request: SmsRequest) -> SmsResult:
        self.requests.append(request)
        if self.status == "sent":
            return SmsResult(
                status="sent",
                provider_message_id="SM456",
                metadata={"provider": "fake"},
            )
        return SmsResult(
            status="failed",
            error_message="simulated failure",
            metadata={"provider": "fake"},
        )


def test_llm_orchestrator_can_complete_store_information_from_root():
    service = LLMConversationOrchestratorService(
        make_settings(),
        policy=QueuePolicy(
            [
                ConversationPolicyDecision(
                    action=PolicyAction.SELECT_OPTION,
                    selected_option="store_information",
                    response_text=(
                        "Sure, the easiest place to check store information is on our website: "
                        "https://demo.example.com/store."
                    ),
                )
            ]
        ),
    )

    result = asyncio.run(
        service.run_turn(
            "store-session",
            "Where is your nearest store?",
            call_context="test",
        )
    )

    assert result.final_outcome == "website"
    assert result.artifacts["website_url"] == "https://demo.example.com/store"
    assert "https://demo.example.com/store" in result.text


def test_llm_orchestrator_keeps_authentication_and_sms_as_explicit_app_actions():
    sms_sender = FakeSmsSender()
    service = LLMConversationOrchestratorService(
        make_settings(),
        sms_sender=sms_sender,
        policy=QueuePolicy(
            [
                ConversationPolicyDecision(
                    action=PolicyAction.SELECT_OPTION,
                    selected_option="registration",
                    response_text="I can help you get registered.",
                ),
                ConversationPolicyDecision(
                    action=PolicyAction.SELECT_OPTION,
                    selected_option="registration_sms_confirmation",
                    response_text="Please send the registration confirmation by SMS.",
                ),
            ]
        ),
    )

    first = asyncio.run(service.run_turn("registration", "I want to sign up", call_context="test"))
    assert first.state_snapshot["pending_auth_field"] == "full_name"
    assert first.active_menu == "authentication"

    second = asyncio.run(service.run_turn("registration", "Chris Example", call_context="test"))
    assert second.state_snapshot["pending_auth_field"] == "phone_number"

    third = asyncio.run(service.run_turn("registration", "(415) 555-2671", call_context="test"))
    assert third.active_menu == "registration_terminal"
    assert third.artifacts["sms_status"] == "sent"
    assert third.artifacts["sms_message_id"] == "SM456"
    assert third.adapter_results["authentication"]["status"] == "needs_sms_confirmation"
    assert third.adapter_results["sms"]["status"] == "sent"
    assert sms_sender.requests[0].recipient_phone_number == "+14155552671"

    fourth = asyncio.run(
        service.run_turn("registration", "send the registration sms", call_context="test")
    )
    assert fourth.final_outcome == "registration_sms_confirmation"


def test_llm_orchestrator_reports_sms_failures_without_claiming_success():
    service = LLMConversationOrchestratorService(
        make_settings(),
        sms_sender=FakeSmsSender(status="failed"),
        policy=QueuePolicy(
            [
                ConversationPolicyDecision(
                    action=PolicyAction.SELECT_OPTION,
                    selected_option="registration",
                    response_text="I can help you get registered.",
                ),
            ]
        ),
    )

    asyncio.run(service.run_turn("registration-failure", "I want to sign up", call_context="test"))
    asyncio.run(service.run_turn("registration-failure", "Chris Example", call_context="test"))
    third = asyncio.run(
        service.run_turn("registration-failure", "(415) 555-2671", call_context="test")
    )

    assert third.active_menu == "registration_terminal"
    assert third.artifacts["sms_status"] == "failed"
    assert third.adapter_results["sms"]["status"] == "failed"
    assert "couldn't send the sms confirmation" in third.text.lower()


def test_llm_orchestrator_reprompts_for_invalid_registration_phone_number():
    service = LLMConversationOrchestratorService(
        make_settings(),
        sms_sender=FakeSmsSender(),
        policy=QueuePolicy(
            [
                ConversationPolicyDecision(
                    action=PolicyAction.SELECT_OPTION,
                    selected_option="registration",
                    response_text="I can help you get registered.",
                ),
            ]
        ),
    )

    asyncio.run(service.run_turn("registration-invalid", "I want to sign up", call_context="test"))
    asyncio.run(service.run_turn("registration-invalid", "Chris Example", call_context="test"))
    third = asyncio.run(
        service.run_turn("registration-invalid", "call me maybe", call_context="test")
    )

    assert third.active_menu == "authentication"
    assert third.state_snapshot["pending_auth_field"] == "phone_number"
    assert "including the country code" in third.text.lower()


def test_llm_orchestrator_uses_genesys_adapter_for_announcements_support():
    service = LLMConversationOrchestratorService(
        make_settings(),
        policy=QueuePolicy(
            [
                ConversationPolicyDecision(
                    action=PolicyAction.SELECT_OPTION,
                    selected_option="announcements",
                    response_text="I can walk you through the latest announcements.",
                ),
                ConversationPolicyDecision(
                    action=PolicyAction.SELECT_OPTION,
                    selected_option="continue",
                    response_text="I'll connect you to support after the announcements.",
                ),
                ConversationPolicyDecision(
                    action=PolicyAction.SELECT_OPTION,
                    selected_option="call_back",
                    response_text="A call back would be best.",
                ),
            ]
        ),
    )

    first = asyncio.run(service.run_turn("announce", "Any announcements?", call_context="test"))
    assert first.active_menu == "announcements_continue"

    second = asyncio.run(service.run_turn("announce", "Yes continue", call_context="test"))
    assert second.active_menu == "announcements_terminal"
    assert second.adapter_results["genesys"]["metadata"]["queue"] == "announcements_queue"

    third = asyncio.run(service.run_turn("announce", "Please call me back", call_context="test"))
    assert third.final_outcome == "call_back"
