import asyncio

from vocode_contact_center.settings import ContactCenterSettings
from vocode_contact_center.voicebot_graph.service import VoicebotGraphService


def make_settings() -> ContactCenterSettings:
    return ContactCenterSettings(
        langchain_provider="openai",
        openai_api_key="openai",
        information_store_website_url="https://demo.example.com/store",
        information_products_pdf_url="https://demo.example.com/products.pdf",
        announcements_message="These are today's announcements.",
        feedback_question_prompt="Do you believe this answered your question? Say yes for Back to Chat or no for contact options.",
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
    assert "let's try again" in third.text.lower()


def test_registration_path_loops_through_customer_input_then_sms_then_terminal_menu():
    service = VoicebotGraphService(make_settings())

    first = asyncio.run(service.run_turn("registration", "I want to register", call_context="test"))
    assert first.state_snapshot["pending_auth_field"] == "full_name"

    second = asyncio.run(service.run_turn("registration", "Chris Example", call_context="test"))
    assert second.state_snapshot["pending_auth_field"] == "phone_number"

    third = asyncio.run(service.run_turn("registration", "+123456789", call_context="test"))
    assert third.active_menu == "registration_terminal"
    assert third.artifacts["sms_status"] == "sent"

    fourth = asyncio.run(
        service.run_turn("registration", "sms confirmation", call_context="test")
    )
    assert fourth.final_outcome == "registration_sms_confirmation"


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
