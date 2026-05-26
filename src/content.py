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
EXTRACTOR_VERSION = "office-v1"
DOCX_CHUNK_CHARS = 2500
MAX_EXCEL_ROWS_PER_SHEET = 200
MAX_EXCEL_COLS_PER_SHEET = 50
SUPPORTED_CONTENT_EXTENSIONS = {".pdf", ".pptx", ".docx", ".xlsx", ".xlsm"}

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
    "workbook",
    "worksheet",
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
        if extension not in SUPPORTED_CONTENT_EXTENSIONS:
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

        extracted = get_or_extract_document_text(
            record=record,
            cache=cache,
            max_units=max_pages_per_file,
            max_chars=max_chars_per_file,
        )
        evidence.append(score_extracted_text(query, extracted))

    save_cache(cache_path, cache)
    return evidence


def get_or_extract_document_text(
    record: dict[str, Any],
    cache: dict[str, Any],
    max_units: int,
    max_chars: int,
) -> dict[str, Any]:
    file_id = str(record.get("file_id"))
    cached = cache.get(file_id)
    cache_key = {
        "extractor_version": EXTRACTOR_VERSION,
        "absolute_path": record.get("absolute_path"),
        "extension": record.get("extension"),
        "file_size_bytes": record.get("file_size_bytes"),
        "max_units": max_units,
        "max_chars": max_chars,
    }

    if cached and cached.get("cache_key") == cache_key:
        cached["cache_hit"] = True
        return cached

    extracted = extract_document_text(
        record,
        max_units=max_units,
        max_chars=max_chars,
    )
    extracted["cache_key"] = cache_key
    extracted["cache_hit"] = False
    cache[file_id] = extracted
    return extracted


def extract_document_text(
    record: dict[str, Any],
    max_units: int,
    max_chars: int,
) -> dict[str, Any]:
    extension = str(record.get("extension", "")).lower()

    if extension == ".pdf":
        return extract_pdf_text(record, max_pages=max_units, max_chars=max_chars)
    if extension == ".pptx":
        return extract_pptx_text(record, max_slides=max_units, max_chars=max_chars)
    if extension == ".docx":
        return extract_docx_text(record, max_chunks=max_units, max_chars=max_chars)
    if extension in {".xlsx", ".xlsm"}:
        return extract_xlsx_text(record, max_sheets=max_units, max_chars=max_chars)

    raise ValueError(f"content extraction for {extension} is not supported yet")


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
        "unit_label": "page",
        "page_count": page_count,
        "inspected_pages": len(pages),
        "char_count": total_chars,
        "pages": pages,
    }


def extract_pptx_text(
    record: dict[str, Any],
    max_slides: int,
    max_chars: int,
) -> dict[str, Any]:
    try:
        from pptx import Presentation
    except ImportError as error:
        raise RuntimeError(
            "PPTX content inspection needs python-pptx. Run: .venv/bin/python -m pip install -r requirements.txt"
        ) from error

    path = Path(str(record.get("absolute_path")))
    presentation = Presentation(path)
    pages: list[dict[str, Any]] = []
    total_chars = 0

    for slide_index, slide in enumerate(presentation.slides, start=1):
        if len(pages) >= max_slides or total_chars >= max_chars:
            break

        text = "\n".join(extract_slide_text(slide))
        remaining_chars = max_chars - total_chars
        text = text[:remaining_chars]
        total_chars += len(text)
        pages.append(
            {
                "page_number": slide_index,
                "text": text,
            }
        )

    return {
        "file_id": record.get("file_id"),
        "filename": record.get("filename"),
        "unit_label": "slide",
        "page_count": len(presentation.slides),
        "inspected_pages": len(pages),
        "char_count": total_chars,
        "pages": pages,
    }


def extract_slide_text(slide: Any) -> list[str]:
    parts: list[str] = []

    for shape in slide.shapes:
        text = getattr(shape, "text", "")
        if text:
            parts.append(text)

        if getattr(shape, "has_table", False):
            for row in shape.table.rows:
                row_values = [
                    cell.text.strip()
                    for cell in row.cells
                    if cell.text and cell.text.strip()
                ]
                if row_values:
                    parts.append(" | ".join(row_values))

    if getattr(slide, "has_notes_slide", False):
        notes_text_frame = getattr(slide.notes_slide, "notes_text_frame", None)
        if notes_text_frame and notes_text_frame.text:
            parts.append("Speaker notes: " + notes_text_frame.text)

    return parts


def extract_docx_text(
    record: dict[str, Any],
    max_chunks: int,
    max_chars: int,
) -> dict[str, Any]:
    try:
        from docx import Document
    except ImportError as error:
        raise RuntimeError(
            "DOCX content inspection needs python-docx. Run: .venv/bin/python -m pip install -r requirements.txt"
        ) from error

    path = Path(str(record.get("absolute_path")))
    document = Document(path)
    parts: list[str] = []

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if text:
            parts.append(text)

    for table in document.tables:
        for row in table.rows:
            row_values = [
                cell.text.strip()
                for cell in row.cells
                if cell.text and cell.text.strip()
            ]
            if row_values:
                parts.append(" | ".join(row_values))

    full_text = "\n".join(parts)[:max_chars]
    chunks = split_text_into_units(full_text, unit_size=DOCX_CHUNK_CHARS)[:max_chunks]
    pages = [
        {
            "page_number": index,
            "text": chunk,
        }
        for index, chunk in enumerate(chunks, start=1)
    ]

    return {
        "file_id": record.get("file_id"),
        "filename": record.get("filename"),
        "unit_label": "chunk",
        "page_count": len(chunks),
        "inspected_pages": len(pages),
        "char_count": sum(len(page["text"]) for page in pages),
        "pages": pages,
    }


def extract_xlsx_text(
    record: dict[str, Any],
    max_sheets: int,
    max_chars: int,
) -> dict[str, Any]:
    try:
        from openpyxl import load_workbook
    except ImportError as error:
        raise RuntimeError(
            "XLSX content inspection needs openpyxl. Run: .venv/bin/python -m pip install -r requirements.txt"
        ) from error

    path = Path(str(record.get("absolute_path")))
    workbook = load_workbook(path, read_only=True, data_only=False)
    pages: list[dict[str, Any]] = []
    total_chars = 0

    for sheet_index, worksheet in enumerate(workbook.worksheets, start=1):
        if len(pages) >= max_sheets or total_chars >= max_chars:
            break

        lines = [f"Sheet: {worksheet.title}"]
        for row in worksheet.iter_rows(
            max_row=min(worksheet.max_row or 0, MAX_EXCEL_ROWS_PER_SHEET),
            max_col=min(worksheet.max_column or 0, MAX_EXCEL_COLS_PER_SHEET),
        ):
            cell_values = []
            for cell in row:
                if cell.value is None:
                    continue
                cell_values.append(f"{cell.coordinate}={cell.value}")
            if cell_values:
                lines.append(" | ".join(cell_values))

            if len("\n".join(lines)) >= max_chars - total_chars:
                break

        text = "\n".join(lines)
        remaining_chars = max_chars - total_chars
        text = text[:remaining_chars]
        total_chars += len(text)
        pages.append(
            {
                "page_number": sheet_index,
                "text": text,
            }
        )

    workbook.close()

    return {
        "file_id": record.get("file_id"),
        "filename": record.get("filename"),
        "unit_label": "sheet",
        "page_count": len(workbook.worksheets),
        "inspected_pages": len(pages),
        "char_count": total_chars,
        "pages": pages,
    }


def split_text_into_units(text: str, unit_size: int) -> list[str]:
    if not text:
        return []

    return [
        text[index : index + unit_size]
        for index in range(0, len(text), unit_size)
    ]


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
            unit_label = str(extracted.get("unit_label", "page"))
            snippet = make_snippet(text, term)
            if snippet:
                snippets.append(f"{unit_label} {page_number}: {snippet}")

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
