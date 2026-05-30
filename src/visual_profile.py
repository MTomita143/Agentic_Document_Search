"""Local visual page profiling for the hybrid visual funnel.

This layer is intentionally cheaper and more deterministic than CLIP/Azure
Vision. It skims candidate PDF pages for human-style clues: visible text,
chart/table-like structure, image density, and dominant colors. CLIP remains
useful, but it becomes one signal instead of the only gatekeeper.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from clip_prefilter import ClipEvidence
from document_store import resolve_document_path
from search import SearchResult, has_cjk, normalize_text, tokenize


COLOR_ALIASES = {
    "青": "blue",
    "青い": "blue",
    "blue": "blue",
    "赤": "red",
    "赤い": "red",
    "red": "red",
    "緑": "green",
    "green": "green",
    "黄色": "yellow",
    "yellow": "yellow",
    "オレンジ": "orange",
    "orange": "orange",
    "紫": "purple",
    "purple": "purple",
    "マゼンタ": "magenta",
    "magenta": "magenta",
    "グレー": "gray",
    "grey": "gray",
    "gray": "gray",
}
CHART_TERMS = {"chart", "charts", "graph", "graphs", "plot", "trend", "グラフ", "チャート", "図表"}
TABLE_TERMS = {"table", "tables", "spreadsheet", "matrix", "表", "テーブル"}
LAYOUT_TERMS = {"layout", "diagram", "visual", "image", "picture", "screenshot", "レイアウト", "図", "画像"}
TEXT_STOPWORDS = {
    "about",
    "also",
    "and",
    "can",
    "document",
    "file",
    "find",
    "have",
    "might",
    "page",
    "pdf",
    "probably",
    "slide",
    "slides",
    "that",
    "the",
    "this",
    "which",
    "with",
}


@dataclass
class PageProfile:
    file_id: str
    filename: str
    page_number: int
    score: float
    observations: list[str]


@dataclass
class QueryVisualProfile:
    terms: list[str]
    colors: set[str]
    wants_chart: bool
    wants_table: bool
    wants_layout: bool


def rank_visual_pages_with_profile(
    query: str,
    results: list[SearchResult],
    max_files: int,
    max_pages_per_file: int,
    top_pages: int,
    selected_pages_per_file: int,
    render_zoom: float = 0.35,
) -> tuple[list[ClipEvidence], dict[str, list[int]]]:
    query_profile = build_query_visual_profile(query)
    all_profiles: list[PageProfile] = []
    page_counts: dict[str, int] = {}
    rendered_counts: dict[str, int] = {}
    errors: list[ClipEvidence] = []

    for result in results[:max_files]:
        record = result.record
        file_id = str(record.get("file_id"))
        filename = str(record.get("filename"))
        extension = str(record.get("extension", "")).lower()

        if extension != ".pdf":
            errors.append(
                ClipEvidence(
                    file_id=file_id,
                    score=0,
                    selected_pages=[],
                    observations=[],
                    rendered_pages=0,
                    page_count=0,
                    cache_hits=0,
                    error=f"local visual skim for {extension} is not supported yet",
                )
            )
            continue

        try:
            page_profiles, page_count = profile_pdf_pages(
                query_profile=query_profile,
                record=record,
                max_pages=max_pages_per_file,
                render_zoom=render_zoom,
            )
        except Exception as error:
            errors.append(
                ClipEvidence(
                    file_id=file_id,
                    score=0,
                    selected_pages=[],
                    observations=[],
                    rendered_pages=0,
                    page_count=0,
                    cache_hits=0,
                    error=str(error),
                )
            )
            continue

        page_counts[file_id] = page_count
        rendered_counts[file_id] = len(page_profiles)
        all_profiles.extend(page_profiles)

    selected_profiles = select_profiles_for_vision(
        all_profiles,
        top_pages=top_pages,
        selected_pages_per_file=selected_pages_per_file,
    )
    selected_by_file: dict[str, list[int]] = {}
    evidence_by_file: dict[str, ClipEvidence] = {}

    for profile in selected_profiles:
        selected_by_file.setdefault(profile.file_id, []).append(profile.page_number)
        observation = (
            f"page {profile.page_number} matched the local visual skim"
            + (": " + "; ".join(profile.observations) if profile.observations else "")
        )
        current = evidence_by_file.get(profile.file_id)
        if current is None:
            evidence_by_file[profile.file_id] = ClipEvidence(
                file_id=profile.file_id,
                score=max(profile.score, 0.0),
                selected_pages=[profile.page_number],
                observations=[observation],
                rendered_pages=rendered_counts.get(profile.file_id, 0),
                page_count=page_counts.get(profile.file_id, 0),
                cache_hits=0,
            )
        else:
            current.score += max(profile.score, 0.0)
            current.selected_pages.append(profile.page_number)
            current.observations.append(observation)

    return list(evidence_by_file.values()) + errors, selected_by_file


def build_query_visual_profile(query: str) -> QueryVisualProfile:
    normalized = normalize_text(query)
    tokens = tokenize(query)
    colors = {
        color
        for term, color in COLOR_ALIASES.items()
        if term in tokens or term in normalized
    }
    wants_chart = any(term in tokens or term in normalized for term in CHART_TERMS)
    wants_table = any(term in tokens or term in normalized for term in TABLE_TERMS)
    wants_layout = any(term in tokens or term in normalized for term in LAYOUT_TERMS)
    terms = [
        term
        for term in tokens
        if (len(term) >= 3 or has_cjk(term)) and term not in TEXT_STOPWORDS
    ][:12]

    return QueryVisualProfile(
        terms=terms,
        colors=colors,
        wants_chart=wants_chart,
        wants_table=wants_table,
        wants_layout=wants_layout,
    )


def profile_pdf_pages(
    query_profile: QueryVisualProfile,
    record: dict[str, Any],
    max_pages: int,
    render_zoom: float,
) -> tuple[list[PageProfile], int]:
    try:
        import fitz
    except ImportError as error:
        raise RuntimeError(
            "Local visual skim needs PyMuPDF. Run: .venv/bin/python -m pip install -r requirements.txt"
        ) from error

    file_id = str(record.get("file_id"))
    filename = str(record.get("filename"))
    path = resolve_document_path(record)
    profiles: list[PageProfile] = []

    with fitz.open(path) as document:
        page_count = document.page_count
        for page_number in choose_spread_pages(page_count, max_pages):
            page = document.load_page(page_number - 1)
            profiles.append(
                profile_page(
                    page=page,
                    file_id=file_id,
                    filename=filename,
                    page_number=page_number,
                    query_profile=query_profile,
                    render_zoom=render_zoom,
                )
            )

    return profiles, page_count


def choose_spread_pages(page_count: int, max_pages: int) -> list[int]:
    if page_count <= 0 or max_pages <= 0:
        return []
    if page_count <= max_pages:
        return list(range(1, page_count + 1))
    if max_pages == 1:
        return [1]

    pages = {1, page_count}
    slots = max_pages - len(pages)
    for index in range(1, slots + 1):
        page = round(1 + index * (page_count - 1) / (slots + 1))
        pages.add(max(1, min(page_count, page)))

    next_page = 2
    while len(pages) < max_pages and next_page <= page_count:
        pages.add(next_page)
        next_page += 1

    return sorted(pages)


def profile_page(
    page: Any,
    file_id: str,
    filename: str,
    page_number: int,
    query_profile: QueryVisualProfile,
    render_zoom: float,
) -> PageProfile:
    text = page.get_text("text") or ""
    normalized_text = normalize_text(text)
    dict_text = page.get_text("dict") or {}
    drawings = safe_count_drawings(page)
    image_blocks = count_image_blocks(dict_text)
    table_count = safe_count_tables(page)
    numeric_lines = count_numeric_lines(text)
    table_like_lines = count_table_like_lines(text)
    color_ratios = estimate_color_ratios(page, render_zoom)

    score = 0.0
    observations: list[str] = []

    matched_terms = [
        term
        for term in query_profile.terms
        if term_matches(normalized_text, term)
    ]
    if matched_terms:
        score += min(24.0, len(matched_terms) * 4.0)
        observations.append("visible text matched " + ", ".join(matched_terms[:4]))

    if query_profile.wants_chart:
        chart_score = min(24.0, drawings * 0.35 + image_blocks * 3.0 + numeric_lines * 1.1)
        if chart_score >= 4.0:
            score += chart_score
            observations.append("chart-like layout")

    if query_profile.wants_table:
        table_score = min(24.0, table_count * 12.0 + table_like_lines * 2.0)
        if table_score >= 3.0:
            score += table_score
            observations.append("table-like structure")

    if query_profile.wants_layout:
        layout_score = min(12.0, drawings * 0.2 + image_blocks * 2.0)
        if layout_score >= 2.0:
            score += layout_score
            observations.append("visual/layout-heavy page")

    for color in sorted(query_profile.colors):
        ratio = color_ratios.get(color, 0.0)
        if ratio >= 0.025:
            score += min(14.0, ratio * 140.0)
            observations.append(f"{color} visual theme")

    if not any([query_profile.wants_chart, query_profile.wants_table, query_profile.wants_layout, query_profile.colors]):
        generic_visual_score = min(8.0, drawings * 0.15 + image_blocks * 1.5)
        if generic_visual_score >= 2.0:
            score += generic_visual_score
            observations.append("visually dense page")

    if not observations and (drawings or image_blocks):
        observations.append("sampled visual page")

    return PageProfile(
        file_id=file_id,
        filename=filename,
        page_number=page_number,
        score=score,
        observations=observations[:4],
    )


def select_profiles_for_vision(
    profiles: list[PageProfile],
    top_pages: int,
    selected_pages_per_file: int,
) -> list[PageProfile]:
    selected: list[PageProfile] = []
    selected_counts: dict[str, int] = {}

    for profile in sorted(profiles, key=lambda item: item.score, reverse=True):
        if len(selected) >= top_pages:
            break
        if selected_counts.get(profile.file_id, 0) >= selected_pages_per_file:
            continue
        selected.append(profile)
        selected_counts[profile.file_id] = selected_counts.get(profile.file_id, 0) + 1

    return selected


def safe_count_drawings(page: Any) -> int:
    try:
        return len(page.get_drawings())
    except Exception:
        return 0


def safe_count_tables(page: Any) -> int:
    try:
        tables = page.find_tables()
    except Exception:
        return 0
    return len(getattr(tables, "tables", []) or [])


def count_image_blocks(text_dict: dict[str, Any]) -> int:
    blocks = text_dict.get("blocks", [])
    return sum(1 for block in blocks if block.get("type") == 1)


def count_numeric_lines(text: str) -> int:
    lines = text.splitlines()
    return sum(1 for line in lines if re.search(r"\d", line) and re.search(r"[%$€¥]|20\d{2}|19\d{2}", line))


def count_table_like_lines(text: str) -> int:
    lines = text.splitlines()
    return sum(
        1
        for line in lines
        if line.count("\t") >= 2
        or line.count("|") >= 2
        or re.search(r"\S+\s{2,}\S+\s{2,}\S+", line)
    )


def estimate_color_ratios(page: Any, render_zoom: float) -> dict[str, float]:
    try:
        import fitz

        pixmap = page.get_pixmap(matrix=fitz.Matrix(render_zoom, render_zoom), alpha=False)
    except Exception:
        return {}

    samples = pixmap.samples
    channels = pixmap.n
    if channels < 3:
        return {}

    pixel_count = max(1, len(samples) // channels)
    stride = max(1, math.ceil(pixel_count / 3500))
    counts = {color: 0 for color in {"blue", "red", "green", "yellow", "orange", "purple", "magenta", "gray"}}
    considered = 0

    for pixel_index in range(0, pixel_count, stride):
        offset = pixel_index * channels
        red, green, blue = samples[offset], samples[offset + 1], samples[offset + 2]
        if is_nearly_white(red, green, blue) or is_nearly_black(red, green, blue):
            continue
        considered += 1
        color = classify_color(red, green, blue)
        if color:
            counts[color] += 1

    if considered == 0:
        return {}

    return {color: count / considered for color, count in counts.items()}


def classify_color(red: int, green: int, blue: int) -> str | None:
    if max(red, green, blue) - min(red, green, blue) <= 18:
        average = (red + green + blue) / 3
        if 70 <= average <= 220:
            return "gray"
        return None

    if blue > 90 and blue > red * 1.18 and blue > green * 0.85:
        return "blue"
    if red > 125 and red > green * 1.22 and red > blue * 1.08:
        return "red"
    if green > 105 and green > red * 1.08 and green > blue * 1.08:
        return "green"
    if red > 145 and green > 115 and blue < 110:
        return "yellow" if green > 135 else "orange"
    if red > 120 and blue > 115 and green < 120:
        return "magenta" if red > 145 else "purple"
    return None


def is_nearly_white(red: int, green: int, blue: int) -> bool:
    return red > 238 and green > 238 and blue > 238


def is_nearly_black(red: int, green: int, blue: int) -> bool:
    return red < 24 and green < 24 and blue < 24


def term_matches(normalized_text: str, term: str) -> bool:
    if has_cjk(term):
        return term in normalized_text
    return re.search(rf"\b{re.escape(term)}[a-z0-9]*\b", normalized_text) is not None
