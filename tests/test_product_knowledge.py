from vocode_contact_center.product_knowledge import ProductKnowledgeService
from vocode_contact_center.settings import ContactCenterSettings


def test_product_knowledge_is_not_configured_when_pypdf_is_missing(monkeypatch):
    monkeypatch.setattr("vocode_contact_center.product_knowledge.PdfReader", None)

    service = ProductKnowledgeService(
        ContactCenterSettings(
            langchain_provider="openai",
            openai_api_key="openai",
            information_products_pdf_path="PRODUCT_PAGE.pdf",
        )
    )

    assert service.is_configured() is False


def test_product_knowledge_fallback_avoids_raw_greek_text_for_english_voice():
    service = ProductKnowledgeService(
        ContactCenterSettings(
            langchain_provider="openai",
            openai_api_key="openai",
            information_products_answer_language="English",
        )
    )

    text = service._extractive_fallback(
        "Τα αποτελέσματα εξετάσεων εκδίδονται εντός δύο εργάσιμων ημερών.",
        "PRODUCT_PAGE.pdf",
    )

    assert "clean english voice answer" in text.lower()
    assert "αποτελέσματα" not in text