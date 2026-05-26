"""
Inspect candidate document pages with Azure AI Vision.

This module renders only selected PDF pages, sends those page images to Azure AI
Vision Image Analysis, and caches the responses. The agent decides when to call
this layer; by default it should be used only for visual-memory queries.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from search import SearchResult, tokenize


DEFAULT_VISUAL_CACHE_PATH = Path("indexes/visual_cache.json")
DEFAULT_VISION_API_VERSION = "2024-02-01"
DEFAULT_VISION_FEATURES = "caption,denseCaptions,tags,read,objects"

VISUAL_SCORE_STOPWORDS = {
    "deck",
    "doc",
    "document",
    "file",
    "image",
    "pdf",
    "picture",
    "presentation",
    "probably",
    "report",
    "screenshot",
    "slide",
    "slides",
    "visual",
}


@dataclass
class VisionConfig:
    endpoint: str
    key: str
    api_version: str
    features: str


@dataclass
class VisualEvidence:
    file_id: str
    score: float
    matched_terms: list[str]
    observations: list[str]
    analyzed_pages: int
    page_count: int
    azure_calls: int
    cache_hits: int
    error: str | None = None


def missing_vision_config() -> list[str]:
    names = []
    endpoint = os.getenv("AZURE_VISION_ENDPOINT") or os.getenv("VISION_ENDPOINT")
    key = os.getenv("AZURE_VISION_KEY") or os.getenv("VISION_KEY")

    if not endpoint:
        names.append("AZURE_VISION_ENDPOINT or VISION_ENDPOINT")
    if not key:
        names.append("AZURE_VISION_KEY or VISION_KEY")

    return names


def load_vision_config() -> VisionConfig:
    endpoint = os.getenv("AZURE_VISION_ENDPOINT") or os.getenv("VISION_ENDPOINT")
    key = os.getenv("AZURE_VISION_KEY") or os.getenv("VISION_KEY")
    api_version = os.getenv("AZURE_VISION_API_VERSION", DEFAULT_VISION_API_VERSION)
    features = os.getenv("AZURE_VISION_FEATURES", DEFAULT_VISION_FEATURES)

    missing = missing_vision_config()
    if missing:
        raise RuntimeError(
            "Azure Vision inspection needs environment variables: "
            + ", ".join(missing)
        )

    return VisionConfig(
        endpoint=str(endpoint),
        key=str(key),
        api_version=api_version,
        features=features,
    )


def inspect_visuals_for_results(
    query: str,
    results: list[SearchResult],
    max_files: int,
    max_pages_per_file: int,
    cache_path: Path = DEFAULT_VISUAL_CACHE_PATH,
    render_zoom: float = 1.5,
    pages_by_file_id: dict[str, list[int]] | None = None,
) -> list[VisualEvidence]:
    config = load_vision_config()
    cache = load_cache(cache_path)
    evidence: list[VisualEvidence] = []

    for result in results[:max_files]:
        record = result.record
        extension = str(record.get("extension", "")).lower()
        if extension != ".pdf":
            evidence.append(
                VisualEvidence(
                    file_id=str(record.get("file_id")),
                    score=0,
                    matched_terms=[],
                    observations=[],
                    analyzed_pages=0,
                    page_count=0,
                    azure_calls=0,
                    cache_hits=0,
                    error=f"visual inspection for {extension} is not supported yet",
                )
            )
            continue

        evidence.append(
            inspect_pdf_visuals(
                query=query,
                record=record,
                config=config,
                cache=cache,
                max_pages=max_pages_per_file,
                render_zoom=render_zoom,
                page_numbers=(
                    pages_by_file_id.get(str(record.get("file_id")))
                    if pages_by_file_id
                    else None
                ),
            )
        )

    save_cache(cache_path, cache)
    return evidence


def inspect_pdf_visuals(
    query: str,
    record: dict[str, Any],
    config: VisionConfig,
    cache: dict[str, Any],
    max_pages: int,
    render_zoom: float,
    page_numbers: list[int] | None = None,
) -> VisualEvidence:
    try:
        import fitz
    except ImportError as error:
        raise RuntimeError(
            "Visual inspection needs PyMuPDF. Run: .venv/bin/python -m pip install -r requirements.txt"
        ) from error

    file_id = str(record.get("file_id"))
    path = Path(str(record.get("absolute_path")))
    analyses: list[dict[str, Any]] = []
    azure_calls = 0
    cache_hits = 0

    with fitz.open(path) as document:
        page_count = document.page_count
        pages_to_analyze = page_numbers or choose_pages(page_count, max_pages)
        pages_to_analyze = [
            page_number
            for page_number in sorted(set(pages_to_analyze))
            if 1 <= page_number <= page_count
        ][:max_pages]

        for page_number in pages_to_analyze:
            cache_key = build_page_cache_key(
                record=record,
                page_number=page_number,
                render_zoom=render_zoom,
                api_version=config.api_version,
                features=config.features,
            )
            cached = cache.get(cache_key)
            if cached:
                cache_hits += 1
                analyses.append(cached)
                continue

            page = document.load_page(page_number - 1)
            image_bytes = render_page_png(page, render_zoom=render_zoom)
            analysis = analyze_image_with_azure_vision(
                image_bytes=image_bytes,
                config=config,
            )
            analysis["file_id"] = file_id
            analysis["filename"] = record.get("filename")
            analysis["page_number"] = page_number
            cache[cache_key] = analysis
            analyses.append(analysis)
            azure_calls += 1

    visual_evidence = score_visual_analyses(
        query=query,
        file_id=file_id,
        analyses=analyses,
        page_count=page_count,
        azure_calls=azure_calls,
        cache_hits=cache_hits,
    )
    return visual_evidence


def choose_pages(page_count: int, max_pages: int) -> list[int]:
    if page_count <= 0:
        return []
    if page_count <= max_pages:
        return list(range(1, page_count + 1))
    if max_pages == 1:
        return [1]

    candidates = [1, max(1, page_count // 2), page_count]
    pages: list[int] = []
    for page_number in candidates:
        if page_number not in pages:
            pages.append(page_number)
        if len(pages) >= max_pages:
            break

    next_page = 2
    while len(pages) < max_pages and next_page <= page_count:
        if next_page not in pages:
            pages.append(next_page)
        next_page += 1

    return sorted(pages)


def render_page_png(page: Any, render_zoom: float) -> bytes:
    try:
        import fitz
    except ImportError as error:
        raise RuntimeError(
            "Visual inspection needs PyMuPDF. Run: .venv/bin/python -m pip install -r requirements.txt"
        ) from error

    matrix = fitz.Matrix(render_zoom, render_zoom)
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    return pixmap.tobytes("png")


def analyze_image_with_azure_vision(
    image_bytes: bytes,
    config: VisionConfig,
) -> dict[str, Any]:
    query = urllib.parse.urlencode(
        {
            "api-version": config.api_version,
            "features": config.features,
        }
    )
    url = f"{config.endpoint.rstrip('/')}/computervision/imageanalysis:analyze?{query}"
    request = urllib.request.Request(
        url,
        data=image_bytes,
        headers={
            "Content-Type": "application/octet-stream",
            "Ocp-Apim-Subscription-Key": config.key,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Azure Vision request failed: {details}") from error


def score_visual_analyses(
    query: str,
    file_id: str,
    analyses: list[dict[str, Any]],
    page_count: int,
    azure_calls: int,
    cache_hits: int,
) -> VisualEvidence:
    query_terms = [
        term
        for term in tokenize(query)
        if len(term) >= 3 and term not in VISUAL_SCORE_STOPWORDS
    ]
    matched_terms: set[str] = set()
    observations: list[str] = []

    for analysis in analyses:
        page_number = int(analysis.get("page_number", 0))
        visual_text = collect_visual_text(analysis)
        normalized = normalize_visual_text(visual_text)

        for term in query_terms:
            if term_matches(normalized, term):
                matched_terms.add(term)

        observation = summarize_analysis(page_number, analysis)
        if observation:
            observations.append(observation)

    unique_observations = dedupe(observations)[:5]
    score = float(len(matched_terms) * 8 + len(unique_observations) * 1.5)

    return VisualEvidence(
        file_id=file_id,
        score=score,
        matched_terms=sorted(matched_terms),
        observations=unique_observations,
        analyzed_pages=len(analyses),
        page_count=page_count,
        azure_calls=azure_calls,
        cache_hits=cache_hits,
    )


def collect_visual_text(analysis: dict[str, Any]) -> str:
    parts: list[str] = []

    caption = analysis.get("captionResult", {}).get("text")
    if caption:
        parts.append(str(caption))

    for dense_caption in analysis.get("denseCaptionsResult", {}).get("values", []):
        text = dense_caption.get("text")
        if text:
            parts.append(str(text))

    for tag in analysis.get("tagsResult", {}).get("values", []):
        name = tag.get("name")
        if name:
            parts.append(str(name))

    for detected_object in analysis.get("objectsResult", {}).get("values", []):
        for tag in detected_object.get("tags", []):
            name = tag.get("name")
            if name:
                parts.append(str(name))

    for block in analysis.get("readResult", {}).get("blocks", []):
        for line in block.get("lines", []):
            text = line.get("text")
            if text:
                parts.append(str(text))

    return " ".join(parts)


def summarize_analysis(page_number: int, analysis: dict[str, Any]) -> str | None:
    caption = analysis.get("captionResult", {}).get("text")
    tags = [
        str(tag.get("name"))
        for tag in analysis.get("tagsResult", {}).get("values", [])
        if tag.get("name")
    ][:5]

    read_lines = []
    for block in analysis.get("readResult", {}).get("blocks", []):
        for line in block.get("lines", []):
            text = line.get("text")
            if text:
                read_lines.append(str(text))
            if len(read_lines) >= 3:
                break
        if len(read_lines) >= 3:
            break

    details = []
    if caption:
        details.append(f"caption: {caption}")
    if tags:
        details.append("tags: " + ", ".join(tags))
    if read_lines:
        details.append("visible text: " + " / ".join(read_lines))

    if not details:
        return None

    return f"page {page_number}: " + "; ".join(details)


def build_page_cache_key(
    record: dict[str, Any],
    page_number: int,
    render_zoom: float,
    api_version: str,
    features: str,
) -> str:
    raw_key = json.dumps(
        {
            "file_id": record.get("file_id"),
            "absolute_path": record.get("absolute_path"),
            "file_size_bytes": record.get("file_size_bytes"),
            "page_number": page_number,
            "render_zoom": render_zoom,
            "api_version": api_version,
            "features": features,
        },
        sort_keys=True,
    )
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def normalize_visual_text(text: str) -> str:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return text.lower()


def term_matches(normalized_text: str, term: str) -> bool:
    pattern = rf"\b{re.escape(term)}[a-z0-9]*\b"
    return re.search(pattern, normalized_text) is not None


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
