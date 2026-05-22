"""
Extract and score text from candidate documents.

This layer is deliberately lazy: it only opens files after metadata search has
selected candidates. Extracted text is cached under indexes/ so repeated local
experiments do not keep re-parsing the same PDFs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from search import SearchResult, tokenize


DEFAULT_CONTENT_CACHE_PATH = Path("indexes/content_cache.json")

CONTENT_SCORE_STOPWORDS = {
    "about",
    "deck",
    "doc",
    "document",
    "file",
    "mentions",
    "pdf",
    "presentation",
    "probably",
    "report",
    "says",
    "slide",
    "slides",
    "talk",
    "talks",
    "topic",
}


@dataclass
class ContentEvidence:
    file_id: str
    score: float
    matched_terms: list[str]
    snippets: list[str]
    inspected_pages: int
    page_count: int
    cache_hit: bool
    error: str | None = None


def inspect_content_for_results(
    query: str,
    results: list[SearchResult],
    max_files: int,
    max_pages_per_file: int,
    max_chars_per_file: int,
    cache_path: Path = DEFAULT_CONTENT_CACHE_PATH,
) -> list[ContentEvidence]:
    cache = load_cache(cache_path)
    evidence: list[ContentEvidence] = []

    for result in results[:max_files]:
        record = result.record
        extension = str(record.get("extension", "")).lower()
        if extension != ".pdf":
            evidence.append(
                ContentEvidence(
                    file_id=str(record.get("file_id")),
                    score=0,
                    matched_terms=[],
                    snippets=[],
                    inspected_pages=0,
                    page_count=0,
                    cache_hit=False,
                    error=f"content extraction for {extension} is not supported yet",
                )
            )
            continue

        extracted = get_or_extract_pdf_text(
            record=record,
            cache=cache,
            max_pages=max_pages_per_file,
            max_chars=max_chars_per_file,
        )
        evidence.append(score_extracted_text(query, extracted))

    save_cache(cache_path, cache)
    return evidence


def get_or_extract_pdf_text(
    record: dict[str, Any],
    cache: dict[str, Any],
    max_pages: int,
    max_chars: int,
) -> dict[str, Any]:
    file_id = str(record.get("file_id"))
    cached = cache.get(file_id)
    cache_key = {
        "absolute_path": record.get("absolute_path"),
        "file_size_bytes": record.get("file_size_bytes"),
        "max_pages": max_pages,
        "max_chars": max_chars,
    }

    if cached and cached.get("cache_key") == cache_key:
        cached["cache_hit"] = True
        return cached

    extracted = extract_pdf_text(record, max_pages=max_pages, max_chars=max_chars)
    extracted["cache_key"] = cache_key
    extracted["cache_hit"] = False
    cache[file_id] = extracted
    return extracted


def extract_pdf_text(
    record: dict[str, Any],
    max_pages: int,
    max_chars: int,
) -> dict[str, Any]:
    try:
        import fitz
    except ImportError as error:
        raise RuntimeError(
            "Content inspection needs PyMuPDF. Run: .venv/bin/python -m pip install -r requirements.txt"
        ) from error

    path = Path(str(record.get("absolute_path")))
    pages: list[dict[str, Any]] = []
    total_chars = 0

    with fitz.open(path) as document:
        page_count = document.page_count
        for page_index in range(min(page_count, max_pages)):
            if total_chars >= max_chars:
                break

            page = document.load_page(page_index)
            text = page.get_text("text")
            remaining_chars = max_chars - total_chars
            text = text[:remaining_chars]
            total_chars += len(text)
            pages.append(
                {
                    "page_number": page_index + 1,
                    "text": text,
                }
            )

    return {
        "file_id": record.get("file_id"),
        "filename": record.get("filename"),
        "page_count": page_count,
        "inspected_pages": len(pages),
        "char_count": total_chars,
        "pages": pages,
    }


def score_extracted_text(query: str, extracted: dict[str, Any]) -> ContentEvidence:
    file_id = str(extracted.get("file_id"))
    query_terms = [
        term
        for term in tokenize(query)
        if len(term) >= 3 and term not in CONTENT_SCORE_STOPWORDS
    ]
    matched_terms: set[str] = set()
    snippets: list[str] = []

    for page in extracted.get("pages", []):
        page_number = int(page.get("page_number", 0))
        text = str(page.get("text", ""))
        normalized_text = normalize_for_content(text)

        for term in query_terms:
            if not term_matches(normalized_text, term):
                continue

            matched_terms.add(term)
            snippet = make_snippet(text, term)
            if snippet:
                snippets.append(f"page {page_number}: {snippet}")

    unique_snippets = dedupe(snippets)[:3]
    score = float(len(matched_terms) * 6 + len(unique_snippets) * 2)

    return ContentEvidence(
        file_id=file_id,
        score=score,
        matched_terms=sorted(matched_terms),
        snippets=unique_snippets,
        inspected_pages=int(extracted.get("inspected_pages", 0)),
        page_count=int(extracted.get("page_count", 0)),
        cache_hit=bool(extracted.get("cache_hit", False)),
    )


def normalize_for_content(text: str) -> str:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return text.lower()


def term_matches(normalized_text: str, term: str) -> bool:
    pattern = rf"\b{re.escape(term)}[a-z0-9]*\b"
    return re.search(pattern, normalized_text) is not None


def make_snippet(text: str, term: str, window: int = 80) -> str | None:
    match = re.search(rf"\b{re.escape(term)}[a-z0-9]*\b", text, re.IGNORECASE)
    if not match:
        return None

    start = max(match.start() - window, 0)
    end = min(match.end() + window, len(text))
    snippet = re.sub(r"\s+", " ", text[start:end]).strip()
    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."
    return snippet


def load_cache(cache_path: Path) -> dict[str, Any]:
    if not cache_path.exists():
        return {}

    with cache_path.open("r", encoding="utf-8") as f:
        loaded = json.load(f)

    if isinstance(loaded, dict):
        return loaded
    return {}


def save_cache(cache_path: Path, cache: dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []

    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)

    return output
