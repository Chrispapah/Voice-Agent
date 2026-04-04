import asyncio

from vocode_contact_center.product_knowledge import ProductKnowledgeAnswer
from vocode_contact_center.settings import ContactCenterSettings
from vocode_contact_center.voicebot_graph.adapters.base import SmsRequest, SmsResult
from vocode_contact_center.voicebot_graph.service import VoicebotGraphService


def make_settings() -> ContactCenterSettings:
    return ContactCenterSettings(
        langchain_provider="openai",
        openai_api_key="openai",
        sms_default_region="US",
        information_store_website_url="https://demo.example.com/store",
        information_products_pdf_url="https://demo.example.com/products.pdf",
        announcements_message="These are today's announcements.",
        feedback_question_prompt="Do you believe this answered your question? Say yes for Back to Chat or no for contact options.",
    )


class FakeSmsSender:
    def __init__(self, *, status: str = "sent"):
        self.status = status
        self.requests: list[SmsRequest] = []

    async def send(self, request: SmsRequest) -> SmsResult:
        self.requests.append(request)
        if self.status == "sent":
            return SmsResult(
                status="sent",
                provider_message_id="SM123",
                metadata={"provider": "fake"},
            )
        return SmsResult(
            status="failed",
            error_message="simulated failure",
            metadata={"provider": "fake"},
        )


class FakeProductInformationService:
    def __init__(self, *, configured: bool = True):
        self.configured = configured
        self.questions: list[str] = []

    def is_configured(self) -> bool:
        return self.configured

    async def answer_question(self, question: str) -> ProductKnowledgeAnswer:
        self.questions.append(question)
        return ProductKnowledgeAnswer(
            text="The product PDF says the savings account includes online access and monthly statements.",
            artifacts={
                "pdf_reference": "https://demo.example.com/products.pdf",
                "product_source_pages": "2",
            },
            found_match=True,
        )


def test_information_path_loops_back_when_change_information_is_selected():
    service = VoicebotGraphService(make_settings())

    first = asyncio.run(
        service.run_turn(
            "info-session",
            "I need information",
            call_context="test",
        )
    )
    assert first.active_menu == "info_selection"

    second = asyncio.run(
        service.run_turn(
            "info-session",
            "other",
            call_context="test",
        )
    )
    assert second.active_menu == "change_information"

    third = asyncio.run(
        service.run_turn(
            "info-session",
            "change information",
            call_context="test",
        )
    )
    assert third.active_menu == "info_selection"
    assert "would you like store information" in third.text.lower()


def test_product_information_path_answers_from_pdf_service_and_stays_active():
    product_service = FakeProductInformationService()
    service = VoicebotGraphService(
        make_settings(),
        product_information_service=product_service,
    )

    first = asyncio.run(
        service.run_turn(
            "product-info",
            "I need information",
            call_context="test",
        )
    )
    assert first.active_menu == "info_selection"

    second = asyncio.run(
        service.run_turn(
            "product-info",
            "products",
            call_context="test",
        )
    )
    assert second.active_menu == "information_products"
    assert "product pdf" in second.text.lower()

    third = asyncio.run(
        service.run_turn(
            "product-info",
            "Tell me about the savings account",
            call_context="test",
        )
    )
    assert third.active_menu == "information_products"
    assert "savings account" in third.text.lower()
    assert third.artifacts["pdf_reference"] == "https://demo.example.com/products.pdf"
    assert third.artifacts["product_source_pages"] == "2"
    assert product_service.questions == ["Tell me about the savings account"]


def test_product_information_falls_back_to_pdf_link_when_pdf_service_is_not_configured():
    service = VoicebotGraphService(
        make_settings(),
        product_information_service=FakeProductInformationService(configured=False),
    )

    asyncio.run(service.run_turn("product-link", "I need information", call_context="test"))
    result = asyncio.run(service.run_turn("product-link", "products", call_context="test"))

    assert result.final_outcome == "pdf"
    assert result.artifacts["pdf_reference"] == "https://demo.example.com/products.pdf"


def test_registration_path_loops_through_customer_input_then_sms_then_terminal_menu():
    sms_sender = FakeSmsSender()
    service = VoicebotGraphService(make_settings(), sms_sender=sms_sender)

    first = asyncio.run(service.run_turn("registration", "I want to register", call_context="test"))
    assert first.state_snapshot["pending_auth_field"] == "full_name"

    second = asyncio.run(service.run_turn("registration", "Chris Example", call_context="test"))
    assert second.state_snapshot["pending_auth_field"] == "phone_number"
    assert second.state_snapshot["conversation_memory"]["full_name"] == "chris example"

    third = asyncio.run(service.run_turn("registration", "(415) 555-2671", call_context="test"))
    assert third.active_menu == "registration_terminal"
    assert third.artifacts["sms_status"] == "sent"
    assert third.artifacts["sms_message_id"] == "SM123"
    assert third.adapter_results["sms"]["status"] == "sent"
    assert sms_sender.requests[0].recipient_phone_number == "+14155552671"
    assert third.state_snapshot["conversation_memory"]["phone_number"] == "+14155552671"

    fourth = asyncio.run(
        service.run_turn("registration", "sms confirmation", call_context="test")
    )
    assert fourth.final_outcome == "registration_sms_confirmation"


def test_registration_sms_failure_is_truthful_in_graph_flow():
    service = VoicebotGraphService(
        make_settings(),
        sms_sender=FakeSmsSender(status="failed"),
    )

    asyncio.run(service.run_turn("registration-failure", "I want to register", call_context="test"))
    asyncio.run(service.run_turn("registration-failure", "Chris Example", call_context="test"))
    third = asyncio.run(
        service.run_turn("registration-failure", "(415) 555-2671", call_context="test")
    )

    assert third.active_menu == "registration_terminal"
    assert third.artifacts["sms_status"] == "failed"
    assert third.adapter_results["sms"]["status"] == "failed"
    assert "couldn't send the sms confirmation" in third.text.lower()


def test_registration_rejects_transcriber_echo_of_bot_prompt_as_name():
    sms_sender = FakeSmsSender()
    service = VoicebotGraphService(make_settings(), sms_sender=sms_sender)
    asyncio.run(service.run_turn("registration-echo", "I want to register", call_context="test"))
    bad = asyncio.run(
        service.run_turn("registration-echo", "I need your first", call_context="test")
    )
    assert bad.state_snapshot["pending_auth_field"] == "full_name"
    assert "full_name" not in bad.state_snapshot.get("collected_data", {})


def test_registration_accepts_split_first_and_last_name_in_two_turns():
    sms_sender = FakeSmsSender()
    service = VoicebotGraphService(make_settings(), sms_sender=sms_sender)
    asyncio.run(service.run_turn("split-name", "I want to register", call_context="test"))
    asyncio.run(
        service.run_turn("split-name", "First name, Christos", call_context="test")
    )
    third = asyncio.run(
        service.run_turn("split-name", "Last name, Pappas", call_context="test")
    )
    assert third.state_snapshot["pending_auth_field"] == "phone_number"
    assert third.state_snapshot["conversation_memory"]["full_name"] == "christos pappas"


def test_registration_accepts_first_and_last_name_in_one_utterance():
    sms_sender = FakeSmsSender()
    service = VoicebotGraphService(make_settings(), sms_sender=sms_sender)
    asyncio.run(service.run_turn("one-utt-name", "I want to register", call_context="test"))
    second = asyncio.run(
        service.run_turn(
            "one-utt-name",
            "First name Jane last name Smith",
            call_context="test",
        )
    )
    assert second.state_snapshot["pending_auth_field"] == "phone_number"
    assert second.state_snapshot["conversation_memory"]["full_name"] == "jane smith"


def test_registration_full_name_step_reprompts_on_other_keyword_instead_of_main_menu():
    service = VoicebotGraphService(make_settings())
    asyncio.run(service.run_turn("reg-other", "I want to register", call_context="test"))
    second = asyncio.run(service.run_turn("reg-other", "other", call_context="test"))
    assert second.state_snapshot["pending_auth_field"] == "full_name"
    assert second.active_menu is None
    assert second.state_snapshot["current_path"] == "interaction"


def test_registration_reprompts_when_full_name_is_greeting_or_filler():
    sms_sender = FakeSmsSender()
    service = VoicebotGraphService(make_settings(), sms_sender=sms_sender)
    asyncio.run(service.run_turn("registration-name", "I want to register", call_context="test"))
    bad = asyncio.run(service.run_turn("registration-name", "Hello?", call_context="test"))
    assert bad.state_snapshot["pending_auth_field"] == "full_name"
    assert "full_name" not in bad.state_snapshot.get("collected_data", {})
    assert "first" in bad.text.lower() and "last" in bad.text.lower()
    good = asyncio.run(service.run_turn("registration-name", "Chris Example", call_context="test"))
    assert good.state_snapshot["conversation_memory"]["full_name"] == "chris example"
    assert good.state_snapshot["pending_auth_field"] == "phone_number"


def test_registration_reprompts_when_phone_number_is_invalid():
    sms_sender = FakeSmsSender()
    service = VoicebotGraphService(make_settings(), sms_sender=sms_sender)

    asyncio.run(service.run_turn("registration-invalid", "I want to register", call_context="test"))
    asyncio.run(service.run_turn("registration-invalid", "Chris Example", call_context="test"))
    third = asyncio.run(
        service.run_turn("registration-invalid", "call me maybe", call_context="test")
    )

    assert third.active_menu is None
    assert third.state_snapshot["pending_auth_field"] == "phone_number"
    assert "including the country code" in third.text.lower()
    assert sms_sender.requests == []


def test_cancel_during_pending_auth_returns_to_root_and_preserves_safe_memory():
    service = VoicebotGraphService(make_settings())

    asyncio.run(service.run_turn("registration-cancel", "I want to register", call_context="test"))
    second = asyncio.run(
        service.run_turn("registration-cancel", "Chris Example", call_context="test")
    )
    assert second.state_snapshot["pending_auth_field"] == "phone_number"

    cancelled = asyncio.run(
        service.run_turn("registration-cancel", "cancel verification", call_context="test")
    )

    assert cancelled.active_menu == "root_intent"
    assert cancelled.final_outcome == "cancelled_to_main_menu"
    assert cancelled.state_snapshot["pending_auth_field"] is None
    assert cancelled.state_snapshot["interaction_context"] is None
    assert cancelled.state_snapshot["current_path"] is None
    assert cancelled.state_snapshot["collected_data"] == {}
    assert cancelled.state_snapshot["auth_attempts"] == 0
    assert cancelled.state_snapshot["conversation_memory"]["full_name"] == "chris example"
    assert "main menu" in cancelled.text.lower()


def test_cancel_after_sms_prompt_returns_to_root_and_keeps_normalized_phone():
    sms_sender = FakeSmsSender()
    service = VoicebotGraphService(make_settings(), sms_sender=sms_sender)

    asyncio.run(service.run_turn("registration-root", "I want to register", call_context="test"))
    asyncio.run(service.run_turn("registration-root", "Chris Example", call_context="test"))
    third = asyncio.run(
        service.run_turn("registration-root", "(415) 555-2671", call_context="test")
    )
    assert third.active_menu == "registration_terminal"

    cancelled = asyncio.run(
        service.run_turn("registration-root", "main menu", call_context="test")
    )

    assert cancelled.active_menu == "root_intent"
    assert cancelled.final_outcome == "cancelled_to_main_menu"
    assert cancelled.state_snapshot["terminal_group"] is None
    assert cancelled.state_snapshot["pending_auth_field"] is None
    assert cancelled.state_snapshot["conversation_memory"]["full_name"] == "chris example"
    assert cancelled.state_snapshot["conversation_memory"]["phone_number"] == "+14155552671"

    restarted = asyncio.run(
        service.run_turn("registration-root", "I want to register", call_context="test")
    )
    assert restarted.state_snapshot["pending_auth_field"] == "full_name"
    assert restarted.state_snapshot["collected_data"] == {}
    assert restarted.state_snapshot["conversation_memory"]["phone_number"] == "+14155552671"


def test_cross_function_request_during_auth_redirects_to_main_menu():
    service = VoicebotGraphService(make_settings())

    asyncio.run(service.run_turn("registration-switch", "I want to register", call_context="test"))
    second = asyncio.run(
        service.run_turn("registration-switch", "Chris Example", call_context="test")
    )
    assert second.state_snapshot["pending_auth_field"] == "phone_number"

    redirected = asyncio.run(
        service.run_turn("registration-switch", "Actually I need announcements", call_context="test")
    )

    assert redirected.active_menu == "root_intent"
    assert redirected.final_outcome == "cancelled_to_main_menu"
    assert redirected.state_snapshot["pending_auth_field"] is None
    assert redirected.state_snapshot["current_path"] is None
    assert redirected.state_snapshot["conversation_memory"]["full_name"] == "chris example"


def test_cross_function_request_from_active_terminal_redirects_to_main_menu():
    sms_sender = FakeSmsSender()
    service = VoicebotGraphService(make_settings(), sms_sender=sms_sender)

    asyncio.run(service.run_turn("announcements-switch", "announcements", call_context="test"))
    second = asyncio.run(
        service.run_turn("announcements-switch", "continue", call_context="test")
    )
    assert second.active_menu == "announcements_terminal"

    redirected = asyncio.run(
        service.run_turn("announcements-switch", "I need information instead", call_context="test")
    )

    assert redirected.active_menu == "root_intent"
    assert redirected.final_outcome == "cancelled_to_main_menu"
    assert redirected.state_snapshot["current_path"] is None


def test_login_failure_routes_to_fallback_terminal_menu():
    service = VoicebotGraphService(make_settings())

    first = asyncio.run(service.run_turn("login", "login", call_context="test"))
    assert first.state_snapshot["pending_auth_field"] == "account_id"

    second = asyncio.run(service.run_turn("login", "invalid-account", call_context="test"))
    assert second.state_snapshot["pending_auth_field"] == "password_hint"

    third = asyncio.run(service.run_turn("login", "secret hint", call_context="test"))
    assert third.active_menu == "fail_terminal"

    fourth = asyncio.run(service.run_turn("login", "communication", call_context="test"))
    assert fourth.final_outcome == "communication"


def test_announcements_path_calls_genesys_then_offers_terminal_choices():
    service = VoicebotGraphService(make_settings())

    first = asyncio.run(service.run_turn("announcements", "announcements", call_context="test"))
    assert first.active_menu == "announcements_continue"

    second = asyncio.run(service.run_turn("announcements", "continue", call_context="test"))
    assert second.active_menu == "announcements_terminal"
    assert second.adapter_results["genesys"]["metadata"]["queue"] == "announcements_queue"

    third = asyncio.run(service.run_turn("announcements", "call back", call_context="test"))
    assert third.final_outcome == "call_back"


def test_feedback_path_supports_back_to_chat_and_contact_routes():
    service = VoicebotGraphService(make_settings())

    first = asyncio.run(service.run_turn("feedback", "feedback", call_context="test"))
    assert first.active_menu == "feedback_question"

    second = asyncio.run(service.run_turn("feedback", "no", call_context="test"))
    assert second.active_menu == "feedback_terminal"
    assert second.adapter_results["genesys"]["metadata"]["queue"] == "feedback_queue"

    third = asyncio.run(service.run_turn("feedback", "contact", call_context="test"))
    assert third.final_outcome == "contact"
