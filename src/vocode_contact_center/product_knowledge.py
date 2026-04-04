from __future__ import annotations

import asyncio
import re
import threading
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse
from urllib.request import urlopen

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
from pypdf import PdfReader

from vocode_contact_center.langchain_support import extract_text_from_langchain_message
from vocode_contact_center.settings import ContactCenterSettings


PRODUCT_QA_SYSTEM_PROMPT = """
You answer product questions for a banking contact center using only the provided PDF excerpts.

Rules:
- Use only the PDF excerpts. Never invent product details.
- If the excerpts do not answer the caller's question, say that you cannot confirm it from the product PDF.
- Keep the answer concise and natural for live voice, at most three short sentences.
- Do not mention that you are looking at chunks or retrieval results.
""".strip()


@dataclass(frozen=True)
class ProductKnowledgeChunk:
    text: str
    pages: tuple[int, ...]


@dataclass(frozen=True)
class ProductKnowledgeAnswer:
    text: str
    artifacts: dict[str, str]
    found_match: bool


class ProductKnowledgeService:
    def __init__(self, settings: ContactCenterSettings) -> None:
        self.settings = settings
        self._cache_lock = threading.Lock()
        self._cached_key: tuple[str, float | None, int] | None = None
        self._cached_chunks: tuple[ProductKnowledgeChunk, ...] = ()

    def is_configured(self) -> bool:
        return self._source_reference() is not None

    def source_reference(self) -> str | None:
        return self._source_reference()

    async def answer_question(self, question: str) -> ProductKnowledgeAnswer:
        reference = self._source_reference()
        if reference is None:
            return ProductKnowledgeAnswer(
                text=(
                    "I don't have a product PDF configured yet, so I can't answer product questions from it right now."
                ),
                artifacts={},
                found_match=False,
            )

        try:
            chunks = await asyncio.to_thread(self._get_chunks)
        except Exception as exc:
            logger.exception("Failed to load product PDF source={} error={}", reference, exc)
            return ProductKnowledgeAnswer(
                text=(
                    "I couldn't open the product PDF just now. "
                    f"You can still review it directly here: {reference}."
                ),
                artifacts={"pdf_reference": reference},
                found_match=False,
            )

        ranked = self._rank_chunks(question, chunks)
        if not ranked:
            return ProductKnowledgeAnswer(
                text=(
                    "I couldn't find that detail in the product PDF. "
                    f"You can review the document here: {reference}."
                ),
                artifacts={"pdf_reference": reference},
                found_match=False,
            )

        selected_chunks = ranked[: max(1, self.settings.information_products_retrieval_chunks)]
        pages = self._format_pages(selected_chunks)
        context = "\n\n".join(
            f"Pages {self._pages_label(chunk.pages)}:\n{chunk.text}"
            for chunk in selected_chunks
        )
        response_text = await self._answer_from_context(
            question=question,
            context=context,
            reference=reference,
        )
        artifacts = {"pdf_reference": reference}
        if pages:
            artifacts["product_source_pages"] = pages
        return ProductKnowledgeAnswer(
            text=response_text,
            artifacts=artifacts,
            found_match=True,
        )

    async def _answer_from_context(self, *, question: str, context: str, reference: str) -> str:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", PRODUCT_QA_SYSTEM_PROMPT),
                ("system", "Product PDF reference: {pdf_reference}"),
                (
                    "human",
                    "Caller question: {question}\n\nRelevant product PDF excerpts:\n{context}",
                ),
            ]
        )
        model = init_chat_model(
            model=self.settings.langchain_model_name,
            model_provider=self.settings.langchain_provider,
            temperature=0,
            max_tokens=max(
                self.settings.langchain_max_tokens,
                self.settings.information_products_answer_max_tokens,
            ),
        )
        chain = prompt | model
        try:
            result = await chain.ainvoke(
                {
                    "pdf_reference": reference,
                    "question": question,
                    "context": context,
                }
            )
            text = extract_text_from_langchain_message(result).strip()
        except Exception as exc:
            logger.exception("Product PDF answer generation failed source={} error={}", reference, exc)
            text = ""

        if text:
            return text
        return self._extractive_fallback(context, reference)

    def _get_chunks(self) -> tuple[ProductKnowledgeChunk, ...]:
        reference = self._source_reference()
        if reference is None:
            return ()

        cache_key = self._cache_key(reference)
        with self._cache_lock:
            if cache_key == self._cached_key and self._cached_chunks:
                return self._cached_chunks

        chunks = tuple(
            self._chunk_pages(
                self._extract_pages(reference),
                chunk_chars=max(400, self.settings.information_products_chunk_chars),
            )
        )
        with self._cache_lock:
            self._cached_key = cache_key
            self._cached_chunks = chunks
        return chunks

    def _cache_key(self, reference: str) -> tuple[str, float | None, int]:
        local_path = self._local_path()
        if local_path is not None:
            try:
                return (
                    str(local_path.resolve()),
                    local_path.stat().st_mtime,
                    self.settings.information_products_chunk_chars,
                )
            except OSError:
                return (str(local_path), None, self.settings.information_products_chunk_chars)
        return (reference, None, self.settings.information_products_chunk_chars)

    def _extract_pages(self, reference: str) -> list[tuple[int, str]]:
        pdf_bytes = self._read_pdf_bytes(reference)
        reader = PdfReader(BytesIO(pdf_bytes))
        pages: list[tuple[int, str]] = []
        for index, page in enumerate(reader.pages, start=1):
            extracted = page.extract_text() or ""
            normalized = self._normalize_whitespace(extracted)
            if normalized:
                pages.append((index, normalized))
        if not pages:
            raise ValueError("The configured product PDF did not contain extractable text.")
        return pages

    def _read_pdf_bytes(self, reference: str) -> bytes:
        local_path = self._local_path()
        if local_path is not None:
            return local_path.read_bytes()

        with urlopen(reference, timeout=15) as response:
            return response.read()

    def _local_path(self) -> Path | None:
        raw_path = (self.settings.information_products_pdf_path or "").strip()
        if not raw_path:
            return None
        return Path(raw_path).expanduser()

    def _source_reference(self) -> str | None:
        local_path = self._local_path()
        if local_path is not None:
            return str(local_path)

        raw_url = (self.settings.information_products_pdf_url or "").strip()
        if not raw_url or raw_url == "https://example.com/products.pdf":
            return None
        parsed = urlparse(raw_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return None
        return raw_url

    def _chunk_pages(
        self,
        pages: Iterable[tuple[int, str]],
        *,
        chunk_chars: int,
    ) -> Iterable[ProductKnowledgeChunk]:
        for page_number, page_text in pages:
            paragraphs = [
                part
                for part in re.split(r"\n{2,}", page_text)
                if self._normalize_whitespace(part)
            ]
            if not paragraphs:
                paragraphs = [page_text]

            current_parts: list[str] = []
            current_size = 0
            for paragraph in paragraphs:
                normalized = self._normalize_whitespace(paragraph)
                if not normalized:
                    continue
                if len(normalized) > chunk_chars:
                    if current_parts:
                        yield ProductKnowledgeChunk(
                            text="\n\n".join(current_parts),
                            pages=(page_number,),
                        )
                        current_parts = []
                        current_size = 0
                    for segment in self._split_long_text(normalized, chunk_chars):
                        yield ProductKnowledgeChunk(text=segment, pages=(page_number,))
                    continue
                if current_parts and current_size + len(normalized) + 2 > chunk_chars:
                    yield ProductKnowledgeChunk(
                        text="\n\n".join(current_parts),
                        pages=(page_number,),
                    )
                    current_parts = []
                    current_size = 0
                current_parts.append(normalized)
                current_size += len(normalized) + 2

            if current_parts:
                yield ProductKnowledgeChunk(text="\n\n".join(current_parts), pages=(page_number,))

    def _rank_chunks(
        self,
        question: str,
        chunks: Iterable[ProductKnowledgeChunk],
    ) -> list[ProductKnowledgeChunk]:
        query_terms = self._tokenize(question)
        if not query_terms:
            return []

        scored: list[tuple[int, ProductKnowledgeChunk]] = []
        for chunk in chunks:
            chunk_terms = self._tokenize(chunk.text)
            overlap = len(query_terms & chunk_terms)
            if overlap == 0:
                continue
            exact_phrases = sum(1 for term in query_terms if term in chunk.text.lower())
            score = overlap * 10 + exact_phrases
            scored.append((score, chunk))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [chunk for _, chunk in scored]

    def _extractive_fallback(self, context: str, reference: str) -> str:
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", self._normalize_whitespace(context))
            if sentence.strip()
        ]
        if sentences:
            preview = " ".join(sentences[:2]).strip()
            if preview:
                return f"Here’s what I found in the product PDF: {preview}"
        return f"I found relevant product details in the PDF, and you can review the full document here: {reference}."

    def _format_pages(self, chunks: Iterable[ProductKnowledgeChunk]) -> str:
        page_numbers = sorted({page for chunk in chunks for page in chunk.pages})
        return ",".join(str(page) for page in page_numbers)

    def _pages_label(self, pages: tuple[int, ...]) -> str:
        if not pages:
            return "unknown"
        if len(pages) == 1:
            return str(pages[0])
        return ", ".join(str(page) for page in pages)

    def _split_long_text(self, text: str, chunk_chars: int) -> Iterable[str]:
        words = text.split()
        current_words: list[str] = []
        current_size = 0
        for word in words:
            addition = len(word) + (1 if current_words else 0)
            if current_words and current_size + addition > chunk_chars:
                yield " ".join(current_words)
                current_words = [word]
                current_size = len(word)
                continue
            current_words.append(word)
            current_size += addition
        if current_words:
            yield " ".join(current_words)

    def _tokenize(self, text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"\b[a-z0-9]{3,}\b", text.lower())
            if token not in {"the", "and", "for", "with", "that", "this", "from"}
        }

    def _normalize_whitespace(self, text: str) -> str:
        lines = [" ".join(line.split()) for line in text.splitlines()]
        non_empty = [line for line in lines if line]
        return "\n".join(non_empty)