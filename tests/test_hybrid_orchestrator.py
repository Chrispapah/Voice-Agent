import asyncio

from vocode_contact_center.orchestration.hybrid_service import (
    HybridConversationOrchestratorService,
    HybridRouteAction,
    HybridRouteDecision,
)
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
        feedback_question_prompt="Would you like to go back to chat or speak to support?",
    )


class QueueRoutePolicy:
    def __init__(self, decisions: list[HybridRouteDecision]):
        self._decisions = list(decisions)

    async def decide(self, **kwargs) -> HybridRouteDecision:
        if not self._decisions:
            raise AssertionError("No queued hybrid routing decision remained for this turn.")
        return self._decisions.pop(0)


class QueueGenericResponder:
    def __init__(self, responses: list[str]):
        self._responses = list(responses)

    async def respond(self, **kwargs) -> str:
        if not self._responses:
            raise AssertionError("No queued generic response remained for this turn.")
        return self._responses.pop(0)


class FakeSmsSender:
    def __init__(self, *, status: str = "sent"):
        self.status = status
        self.requests: list[SmsRequest] = []

    async def send(self, request: SmsRequest) -> SmsResult:
        self.requests.append(request)
        if self.status == "sent":
            return SmsResult(
                status="sent",
                provider_message_id="SM999",
                metadata={"provider": "fake"},
            )
        return SmsResult(
            status="failed",
            error_message="simulated failure",
            metadata={"provider": "fake"},
        )


class FakeProductInformationService:
    def __init__(self):
        self.questions: list[str] = []

    def is_configured(self) -> bool:
        return True

    async def answer_question(self, question: str) -> ProductKnowledgeAnswer:
        self.questions.append(question)
        return ProductKnowledgeAnswer(
            text="The PDF says PRINCE2 results are usually issued within two business days.",
            artifacts={"pdf_reference": "https://demo.example.com/products.pdf"},
            found_match=True,
        )


def test_hybrid_orchestrator_answers_generic_questions_without_graph_activation():
    service = HybridConversationOrchestratorService(
        make_settings(),
        route_policy=QueueRoutePolicy(
            [HybridRouteDecision(action=HybridRouteAction.ANSWER_DIRECTLY)]
        ),
        generic_responder=QueueGenericResponder(["We can help with general support questions."]),
    )

    result = asyncio.run(
        service.run_turn(
            "generic-session",
            "What kind of help can you provide?",
            call_context="test",
        )
    )

    assert result.text == "We can help with general support questions."
    assert result.active_menu is None
    assert result.state_snapshot["hybrid_mode"] == "generic"


def test_hybrid_orchestrator_enters_and_continues_graph_flow():
    sms_sender = FakeSmsSender()
    service = HybridConversationOrchestratorService(
        make_settings(),
        sms_sender=sms_sender,
        route_policy=QueueRoutePolicy(
            [
                HybridRouteDecision(action=HybridRouteAction.ENTER_GRAPH_FLOW),
                HybridRouteDecision(action=HybridRouteAction.CONTINUE_GRAPH),
                HybridRouteDecision(action=HybridRouteAction.CONTINUE_GRAPH),
            ]
        ),
        generic_responder=QueueGenericResponder([]),
    )

    first = asyncio.run(
        service.run_turn("registration", "I want to register", call_context="test")
    )
    assert first.active_menu == "authentication"
    assert first.state_snapshot["pending_auth_field"] == "full_name"
    assert first.state_snapshot["hybrid_mode"] == "graph"

    second = asyncio.run(
        service.run_turn("registration", "Chris Example", call_context="test")
    )
    assert second.active_menu == "authentication"
    assert second.state_snapshot["pending_auth_field"] == "phone_number"
    assert second.state_snapshot["hybrid_mode"] == "graph"

    third = asyncio.run(
        service.run_turn("registration", "(415) 555-2671", call_context="test")
    )
    assert third.active_menu == "authentication"
    assert third.state_snapshot["pending_auth_field"] == "confirmation_code"
    assert third.state_snapshot["hybrid_mode"] == "graph"
    assert third.artifacts["sms_status"] == "sent"
    assert third.artifacts["sms_message_id"] == "SM999"
    assert sms_sender.requests[0].recipient_phone_number == "+14155552671"


def test_hybrid_orchestrator_can_escape_from_graph_and_resume_generic_mode():
    service = HybridConversationOrchestratorService(
        make_settings(),
        route_policy=QueueRoutePolicy(
            [
                HybridRouteDecision(action=HybridRouteAction.ENTER_GRAPH_FLOW),
                HybridRouteDecision(
                    action=HybridRouteAction.ESCAPE_TO_GENERIC,
                    response_text="Sure, we can step out of that and talk generally.",
                ),
                HybridRouteDecision(action=HybridRouteAction.ANSWER_DIRECTLY),
            ]
        ),
        generic_responder=QueueGenericResponder(["We help with account support and information."]),
    )

    first = asyncio.run(
        service.run_turn("escape-session", "I want to register", call_context="test")
    )
    assert first.active_menu == "authentication"
    assert first.state_snapshot["hybrid_mode"] == "graph"

    second = asyncio.run(
        service.run_turn("escape-session", "Never mind, go back", call_context="test")
    )
    assert "step out of that" in second.text.lower()
    assert second.active_menu is None
    assert second.state_snapshot["hybrid_mode"] == "generic"

    third = asyncio.run(
        service.run_turn("escape-session", "What else can you help with?", call_context="test")
    )
    assert third.text == "We help with account support and information."
    assert third.state_snapshot["hybrid_mode"] == "generic"


def test_hybrid_sticky_overrides_answer_directly_during_full_name_collection():
    sms_sender = FakeSmsSender()
    service = HybridConversationOrchestratorService(
        make_settings(),
        sms_sender=sms_sender,
        route_policy=QueueRoutePolicy(
            [
                HybridRouteDecision(action=HybridRouteAction.ENTER_GRAPH_FLOW),
                HybridRouteDecision(action=HybridRouteAction.ANSWER_DIRECTLY),
            ]
        ),
        generic_responder=QueueGenericResponder(["This generic line should not run."]),
    )

    first = asyncio.run(service.run_turn("sticky-name", "I want to register", call_context="test"))
    assert first.state_snapshot["pending_auth_field"] == "full_name"
    assert first.state_snapshot["hybrid_mode"] == "graph"

    second = asyncio.run(
        service.run_turn("sticky-name", "Jane Example Person", call_context="test")
    )
    assert second.state_snapshot["hybrid_mode"] == "graph"
    assert second.state_snapshot["pending_auth_field"] == "phone_number"
    assert "This generic line" not in second.text


def test_hybrid_orchestrator_can_answer_directly_while_graph_is_active():
    service = HybridConversationOrchestratorService(
        make_settings(),
        route_policy=QueueRoutePolicy(
            [
                HybridRouteDecision(action=HybridRouteAction.ENTER_GRAPH_FLOW),
                HybridRouteDecision(action=HybridRouteAction.ANSWER_DIRECTLY),
            ]
        ),
        generic_responder=QueueGenericResponder(["PeopleCert.org lists support hours on the contact page."]),
    )

    first = asyncio.run(service.run_turn("mixed-session", "login", call_context="test"))
    assert first.active_menu == "authentication"
    assert first.state_snapshot["hybrid_mode"] == "graph"

    second = asyncio.run(
        service.run_turn("mixed-session", "Actually, what are your support hours?", call_context="test")
    )
    assert second.text == "PeopleCert.org lists support hours on the contact page."
    assert second.active_menu is None
    assert second.state_snapshot["hybrid_mode"] == "generic"


def test_hybrid_orchestrator_returns_to_generic_mode_after_graph_completion():
    service = HybridConversationOrchestratorService(
        make_settings(),
        route_policy=QueueRoutePolicy(
            [
                HybridRouteDecision(action=HybridRouteAction.ENTER_GRAPH_FLOW),
                HybridRouteDecision(action=HybridRouteAction.CONTINUE_GRAPH),
                HybridRouteDecision(action=HybridRouteAction.ANSWER_DIRECTLY),
            ]
        ),
        generic_responder=QueueGenericResponder(
            ["We help with PeopleCert exams, your account, and announcements."]
        ),
    )

    first = asyncio.run(service.run_turn("info-session", "I need information", call_context="test"))
    assert first.active_menu == "info_selection"
    assert first.state_snapshot["hybrid_mode"] == "graph"

    second = asyncio.run(service.run_turn("info-session", "store", call_context="test"))
    assert second.final_outcome == "website"
    assert second.state_snapshot["hybrid_mode"] == "generic"
    assert second.artifacts["website_url"] == "https://demo.example.com/store"

    third = asyncio.run(
        service.run_turn("info-session", "What services do you offer?", call_context="test")
    )
    assert third.text == "We help with PeopleCert exams, your account, and announcements."
    assert third.state_snapshot["hybrid_mode"] == "generic"


def test_hybrid_product_information_menu_stays_in_graph_when_router_tries_direct_answer():
    product_service = FakeProductInformationService()
    graph_service = VoicebotGraphService(
        make_settings(),
        product_information_service=product_service,
    )
    service = HybridConversationOrchestratorService(
        make_settings(),
        graph_service=graph_service,
        route_policy=QueueRoutePolicy(
            [
                HybridRouteDecision(action=HybridRouteAction.ENTER_GRAPH_FLOW),
                HybridRouteDecision(action=HybridRouteAction.CONTINUE_GRAPH),
                HybridRouteDecision(action=HybridRouteAction.ANSWER_DIRECTLY),
            ]
        ),
        generic_responder=QueueGenericResponder(["This should not be used."]),
    )

    first = asyncio.run(service.run_turn("product-sticky", "I need information", call_context="test"))
    assert first.state_snapshot["hybrid_mode"] == "graph"

    second = asyncio.run(service.run_turn("product-sticky", "products", call_context="test"))
    assert second.active_menu == "information_products"
    assert second.state_snapshot["hybrid_mode"] == "graph"

    third = asyncio.run(
        service.run_turn("product-sticky", "When are PRINCE2 results ready?", call_context="test")
    )
    assert third.active_menu == "information_products"
    assert third.state_snapshot["hybrid_mode"] == "graph"
    assert "prince2" in third.text.lower() or "business" in third.text.lower()
    assert product_service.questions == ["When are PRINCE2 results ready?"]
