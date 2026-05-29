"""
Run the visible agentic document search flow.

This MVP agent is intentionally lightweight. It shows how the system thinks,
but only uses file-level metadata unless the user explicitly chooses a mode
that calls Azure OpenAI for metadata reranking.

Usage:
    python3 src/agent.py --query "find the investor deck" --mode local
    .venv/bin/python src/agent.py --query "find the investor deck" --mode llm
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from azure_ai_search import (
    missing_azure_ai_search_config,
    search_azure_ai_search,
)
from clip_prefilter import (
    DEFAULT_CLIP_CACHE_PATH,
    ClipEvidence,
    rank_visual_pages_with_clip,
)
from content import (
    DEFAULT_CONTENT_CACHE_PATH,
    ContentEvidence,
    inspect_content_for_results,
)
from content_reranker import rerank_with_content_azure_openai
from search import (
    DEFAULT_INDEX_PATH,
    SearchResult,
    dedupe,
    load_environment,
    load_index,
    normalize_text,
    search_llm,
    search_local,
    serialize_results,
    tokenize,
)
from search_memory import (
    DEFAULT_SEARCH_MEMORY_PATH,
    apply_memory_evidence,
    find_memory_candidates,
    remember_search,
)
from semantic_kernel_auto import SearchStrategyPlan, plan_search_strategy
from translator import QueryTranslation, expand_query_with_translator
from visual import (
    DEFAULT_VISUAL_CACHE_PATH,
    VisualEvidence,
    inspect_visuals_for_results,
    missing_vision_config,
)
from visual_worker_client import (
    missing_visual_worker_config,
    rank_visual_pages_with_worker,
)


VISUAL_CAPABLE_EXTENSIONS = {".pdf"}
TEXT_HEAVY_EXTENSIONS = {".doc", ".docx", ".xls", ".xlsm", ".xlsx"}

VISUAL_TERMS = {
    "グラフ",
    "スクショ",
    "スクリーンショット",
    "チャート",
    "テーブル",
    "レイアウト",
    "画像",
    "写真",
    "青",
    "青い",
    "青の",
    "図",
    "図表",
    "表",
    "blue",
    "chart",
    "diagram",
    "graph",
    "image",
    "layout",
    "picture",
    "screenshot",
    "table",
    "visual",
}

CONTENT_TERMS = {
    "について",
    "トピック",
    "内容",
    "分析",
    "書いて",
    "調査",
    "説明",
    "about",
    "analysis",
    "content",
    "explains",
    "mentions",
    "research",
    "says",
    "talks",
    "topic",
}

FILE_TYPE_TERMS = {
    "スライド",
    "プレゼン",
    "レポート",
    "報告",
    "報告書",
    "文書",
    "資料",
    "deck",
    "doc",
    "document",
    "pdf",
    "presentation",
    "report",
    "slide",
    "slides",
}

PATH_TERMS = {
    "directory",
    "folder",
    "path",
    "フォルダ",
    "フォルダー",
    "場所",
    "階層",
}


@dataclass
class QueryUnderstanding:
    query: str
    metadata_clues: list[str]
    content_clues: list[str]
    visual_clues: list[str]
    should_inspect_content: bool
    should_inspect_visuals: bool


@dataclass
class AgentStep:
    name: str
    status: str
    detail: str


@dataclass
class AgentResponse:
    query_understanding: QueryUnderstanding
    steps: list[AgentStep]
    results: list[SearchResult]
    evidence_scope: str


StepCallback = Callable[[list[AgentStep], str | None], None]


def understand_query(query: str) -> QueryUnderstanding:
    tokens = set(tokenize(query))
    normalized_query = normalize_text(query)

    visual_clues = sorted(collect_clues(tokens, normalized_query, VISUAL_TERMS))
    content_clues = sorted(collect_clues(tokens, normalized_query, CONTENT_TERMS))
    metadata_clues = sorted(
        collect_clues(tokens, normalized_query, FILE_TYPE_TERMS | PATH_TERMS)
    )

    should_inspect_visuals = bool(visual_clues)
    should_inspect_content = bool(content_clues or should_inspect_visuals)

    return QueryUnderstanding(
        query=query,
        metadata_clues=metadata_clues,
        content_clues=content_clues,
        visual_clues=visual_clues,
        should_inspect_content=should_inspect_content,
        should_inspect_visuals=should_inspect_visuals,
    )


def collect_clues(
    tokens: set[str],
    normalized_query: str,
    clue_terms: set[str],
) -> set[str]:
    clues = set(tokens & clue_terms)
    clues.update(
        term
        for term in clue_terms
        if term and term in normalized_query
    )
    return clues


def filter_records_by_extension(
    records: list[dict[str, Any]],
    allowed_extensions: set[str] | None,
) -> list[dict[str, Any]]:
    if allowed_extensions is None:
        return records

    normalized_extensions = normalize_extensions(allowed_extensions)
    return [
        record
        for record in records
        if str(record.get("extension", "")).lower() in normalized_extensions
    ]


def normalize_extensions(extensions: set[str]) -> set[str]:
    return {
        extension if extension.startswith(".") else f".{extension}"
        for extension in (value.lower() for value in extensions)
    }


def selected_extensions_support_visuals(
    allowed_extensions: set[str] | None,
) -> bool:
    if allowed_extensions is None:
        return True
    return bool(normalize_extensions(allowed_extensions) & VISUAL_CAPABLE_EXTENSIONS)


def selected_extensions_are_text_heavy(
    allowed_extensions: set[str] | None,
) -> bool:
    if not allowed_extensions:
        return False
    normalized_extensions = normalize_extensions(allowed_extensions)
    return normalized_extensions <= TEXT_HEAVY_EXTENSIONS


def run_agent(
    query: str,
    index_path: Path,
    mode: str,
    top_k: int,
    candidate_pool_size: int,
    content_mode: str,
    translator_mode: str,
    max_inspected_files: int,
    max_pages_per_file: int,
    max_chars_per_file: int,
    content_cache_path: Path,
    visual_mode: str,
    visual_prefilter: str,
    max_visual_files: int,
    max_visual_pages_per_file: int,
    max_clip_pages: int,
    visual_cache_path: Path,
    clip_cache_path: Path,
    orchestration_mode: str = "manual",
    azure_ai_search_mode: str = "never",
    search_memory_path: Path = DEFAULT_SEARCH_MEMORY_PATH,
    allowed_extensions: set[str] | None = None,
    step_callback: StepCallback | None = None,
) -> AgentResponse:
    records = load_index(index_path)
    steps: list[AgentStep] = []
    emit_progress(steps, step_callback, "File Type Filter")
    records = filter_records_by_extension(records, allowed_extensions)
    if allowed_extensions is not None:
        add_step(steps, make_file_type_filter_step(records, allowed_extensions), step_callback)

    if orchestration_mode == "semantic-kernel":
        emit_progress(steps, step_callback, "Semantic Kernel Auto Mode")
        strategy_plan = plan_search_strategy(
            query=query,
            azure_ai_search_available=not missing_azure_ai_search_config(),
        )
        mode = strategy_plan.mode
        content_mode = strategy_plan.content_mode
        translator_mode = strategy_plan.translator_mode
        visual_mode = strategy_plan.visual_mode
        visual_prefilter = strategy_plan.visual_prefilter
        azure_ai_search_mode = strategy_plan.azure_ai_search_mode
        add_step(steps, make_strategy_step(strategy_plan), step_callback)

    if not selected_extensions_support_visuals(allowed_extensions):
        visual_mode = "never"
        visual_prefilter = "none"
    if (
        orchestration_mode == "semantic-kernel"
        and selected_extensions_are_text_heavy(allowed_extensions)
    ):
        mode = "llm"
        content_mode = "auto"

    if mode == "llm" and candidate_pool_size < top_k:
        candidate_pool_size = top_k
    if (
        visual_prefilter == "clip"
        and max_clip_pages < max_visual_files * max_visual_pages_per_file
    ):
        max_clip_pages = max_visual_files * max_visual_pages_per_file

    emit_progress(steps, step_callback, "Translate Query")
    translation = prepare_query_translation(query, translator_mode)
    search_query = translation.expanded_query
    query_understanding = understand_query(search_query)
    query_understanding.query = query

    translation_step = make_translation_step(translation, translator_mode)
    if translation_step:
        add_step(steps, translation_step, step_callback)

    emit_progress(steps, step_callback, "Understand Query")
    add_step(
        steps,
        AgentStep(
            name="Understand Query",
            status="done",
            detail=format_understanding_detail(query_understanding),
        ),
        step_callback,
    )

    emit_progress(steps, step_callback, "Metadata Search")
    if mode == "local" or not records:
        results = search_local(search_query, records, top_k=top_k)
        evidence_scope = "metadata only"
        add_step(
            steps,
            AgentStep(
                name="Metadata Search",
                status="done",
                detail=(
                    f"Searched {len(records)} file metadata records with local "
                    f"keyword and fuzzy matching. Returned top {len(results)}."
                ),
            ),
            step_callback,
        )
    else:
        try:
            results = search_llm(
                search_query,
                records,
                top_k=top_k,
                candidate_pool_size=candidate_pool_size,
            )
            evidence_scope = "metadata only with Azure OpenAI reranking"
            add_step(
                steps,
                AgentStep(
                    name="Metadata Search + LLM Rerank",
                    status="done",
                    detail=(
                        f"Used local metadata search to create a candidate pool, "
                        f"then reranked metadata only with Azure OpenAI. Returned "
                        f"top {len(results)}."
                    ),
                ),
                step_callback,
            )
        except Exception as error:
            results = search_local(search_query, records, top_k=top_k)
            evidence_scope = "metadata only"
            add_step(
                steps,
                AgentStep(
                    name="Metadata Search + LLM Rerank",
                    status="deferred",
                    detail=(
                        "Azure OpenAI reranking could not run, so the agent used "
                        f"local metadata search instead. {error}"
                    ),
                ),
                step_callback,
            )

    emit_progress(steps, step_callback, "Search Memory")
    memory_candidates = find_memory_candidates(
        query=search_query,
        records=records,
        top_k=max(top_k, candidate_pool_size),
        cache_path=search_memory_path,
    )
    if memory_candidates:
        results = merge_search_results(results, memory_candidates)
        evidence_scope = f"{evidence_scope} + search memory"
        add_step(steps, make_memory_step(memory_candidates), step_callback)
    else:
        memory_evidence = apply_memory_evidence(
            query=search_query,
            results=results,
            cache_path=search_memory_path,
        )
        if memory_evidence:
            results = sorted(results, key=lambda result: result.score, reverse=True)

    emit_progress(steps, step_callback, "Azure AI Search")
    azure_search_results = run_optional_azure_ai_search(
        query=search_query,
        records=records,
        top_k=max(top_k, candidate_pool_size),
        azure_ai_search_mode=azure_ai_search_mode,
        steps=steps,
    )
    emit_progress(steps, step_callback, "Inspect Content")
    if azure_search_results:
        results = merge_search_results(results, azure_search_results)
        results = sorted(results, key=lambda result: result.score, reverse=True)
        evidence_scope = f"{evidence_scope} + Azure AI Search candidates"

    if should_run_content_inspection(query_understanding, results, content_mode):
        if not results:
            results = search_local(
                search_query,
                records,
                top_k=max_inspected_files,
                include_zero_scores=True,
            )
        content_evidence = inspect_content_for_results(
            query=search_query,
            results=results,
            max_files=max_inspected_files,
            max_pages_per_file=max_pages_per_file,
            max_chars_per_file=max_chars_per_file,
            cache_path=content_cache_path,
        )
        apply_content_evidence(results, content_evidence)
        results = sorted(results, key=lambda result: result.score, reverse=True)
        evidence_scope = f"{evidence_scope} + extracted candidate text"
        add_step(steps, make_content_step(content_evidence), step_callback)
        if should_run_content_llm_rerank(mode, content_evidence):
            emit_progress(steps, step_callback, "Content LLM Rerank")
            try:
                results = rerank_with_content_azure_openai(
                    query=search_query,
                    results=results,
                    content_evidence=content_evidence,
                    top_k=top_k,
                )
                evidence_scope = f"{evidence_scope} + deep Azure OpenAI content reranking"
                add_step(steps, make_content_llm_step(results), step_callback)
            except Exception as error:
                results = results[:top_k]
                add_step(
                    steps,
                    make_content_llm_deferred_step(str(error)),
                    step_callback,
                )
        else:
            results = results[:top_k]
    else:
        add_step(
            steps,
            make_content_skipped_step(query_understanding, results, content_mode),
            step_callback,
        )

    selected_visual_pages: dict[str, list[int]] | None = None
    clip_prefilter_failed = False

    emit_progress(steps, step_callback, "CLIP Visual Prefilter")
    if should_run_clip_prefilter(query_understanding, results, visual_prefilter):
        try:
            if missing_visual_worker_config():
                clip_evidence, selected_visual_pages = rank_visual_pages_with_clip(
                    query=search_query,
                    results=results,
                    max_files=max_visual_files,
                    max_pages_per_file=max_visual_pages_per_file,
                    top_pages=max_clip_pages,
                    cache_path=clip_cache_path,
                )
                evidence_scope = f"{evidence_scope} + local CLIP page prefilter"
            else:
                clip_evidence, selected_visual_pages = rank_visual_pages_with_worker(
                    query=search_query,
                    results=results,
                    max_files=max_visual_files,
                    max_pages_per_file=max_visual_pages_per_file,
                    top_pages=max_clip_pages,
                )
                evidence_scope = f"{evidence_scope} + GPU VM CLIP page prefilter"

            apply_clip_evidence(results, clip_evidence)
            results = sorted(results, key=lambda result: result.score, reverse=True)[:top_k]
            add_step(steps, make_clip_step(clip_evidence, max_clip_pages), step_callback)
        except RuntimeError as error:
            clip_prefilter_failed = True
            add_step(steps, make_clip_error_step(str(error)), step_callback)
    elif query_understanding.should_inspect_visuals:
        add_step(steps, make_clip_skipped_step(visual_prefilter), step_callback)

    emit_progress(steps, step_callback, "Inspect Visuals")
    if should_run_visual_inspection(query_understanding, results, visual_mode):
        if clip_prefilter_failed and visual_prefilter == "clip":
            add_step(steps, make_visual_blocked_by_clip_step(), step_callback)
        else:
            missing_visual_config = missing_vision_config()
            if missing_visual_config:
                add_step(
                    steps,
                    make_visual_missing_config_step(missing_visual_config),
                    step_callback,
                )
            else:
                visual_evidence = inspect_visuals_for_results(
                    query=search_query,
                    results=results,
                    max_files=max_visual_files,
                    max_pages_per_file=max_visual_pages_per_file,
                    cache_path=visual_cache_path,
                    pages_by_file_id=selected_visual_pages,
                )
                apply_visual_evidence(results, visual_evidence)
                results = sorted(results, key=lambda result: result.score, reverse=True)[:top_k]
                evidence_scope = f"{evidence_scope} + Azure Vision page analysis"
                add_step(steps, make_visual_step(visual_evidence), step_callback)
    elif query_understanding.should_inspect_visuals:
        add_step(steps, make_visual_skipped_step(visual_mode), step_callback)

    emit_progress(steps, step_callback, "Return Answer")
    results = sorted(results, key=lambda result: result.score, reverse=True)[:top_k]
    normalize_result_scores(results)
    add_step(steps, make_answer_step(results), step_callback)
    remember_search(query, results, search_memory_path)

    return AgentResponse(
        query_understanding=query_understanding,
        steps=steps,
        results=results,
        evidence_scope=evidence_scope,
    )


def add_step(
    steps: list[AgentStep],
    step: AgentStep,
    step_callback: StepCallback | None,
) -> None:
    steps.append(step)
    emit_progress(steps, step_callback, None)


def emit_progress(
    steps: list[AgentStep],
    step_callback: StepCallback | None,
    current_step: str | None,
) -> None:
    if step_callback:
        step_callback(list(steps), current_step)


def prepare_query_translation(query: str, translator_mode: str) -> QueryTranslation:
    if translator_mode == "never":
        return QueryTranslation(
            original_query=query,
            expanded_query=query,
            translated_query=None,
            detected_language=None,
            used=False,
        )

    return expand_query_with_translator(query)


def make_translation_step(
    translation: QueryTranslation,
    translator_mode: str,
) -> AgentStep | None:
    if translator_mode == "never":
        return None

    if translation.used:
        detail = f"Expanded the query with Azure Translator: {translation.translated_query}"
        if translation.detected_language:
            detail += f" (detected {translation.detected_language})"
        return AgentStep(
            name="Translate Query",
            status="done",
            detail=detail,
        )

    if translation.error:
        return AgentStep(
            name="Translate Query",
            status="deferred",
            detail=translation.error,
        )

    return AgentStep(
        name="Translate Query",
        status="not needed",
        detail="No Japanese query expansion was needed for this search.",
    )


def make_strategy_step(strategy_plan: SearchStrategyPlan) -> AgentStep:
    status = "done" if strategy_plan.used_semantic_kernel else "fallback"
    detail = strategy_plan.rationale
    if strategy_plan.error:
        detail += " Semantic Kernel note: " + strategy_plan.error
    detail += (
        f" Selected: metadata={strategy_plan.mode}, "
        f"text={strategy_plan.content_mode}, "
        f"translator={strategy_plan.translator_mode}, "
        f"visual={strategy_plan.visual_mode}, "
        f"CLIP={strategy_plan.visual_prefilter}, "
        f"Azure AI Search={strategy_plan.azure_ai_search_mode}."
    )
    return AgentStep(
        name="Semantic Kernel Auto Mode",
        status=status,
        detail=detail,
    )


def make_file_type_filter_step(
    records: list[dict[str, Any]],
    allowed_extensions: set[str],
) -> AgentStep:
    extensions = ", ".join(sorted(normalize_extensions(allowed_extensions)))
    return AgentStep(
        name="File Type Filter",
        status="done",
        detail=f"Limited search to {len(records)} files with extensions: {extensions}.",
    )


def make_memory_step(memory_candidates: list[SearchResult]) -> AgentStep:
    filenames = [
        str(result.record.get("filename"))
        for result in memory_candidates[:3]
    ]
    return AgentStep(
        name="Search Memory",
        status="done",
        detail=(
            "Used previous search sessions as a lightweight memory signal. "
            "Recalled: " + ", ".join(filenames)
        ),
    )


def run_optional_azure_ai_search(
    query: str,
    records: list[dict[str, Any]],
    top_k: int,
    azure_ai_search_mode: str,
    steps: list[AgentStep],
) -> list[SearchResult]:
    if azure_ai_search_mode == "never":
        return []

    missing = missing_azure_ai_search_config()
    if missing:
        steps.append(
            AgentStep(
                name="Azure AI Search",
                status="deferred",
                detail=(
                    "Azure AI Search is available as an optional retrieval tool, "
                    "but it is not configured yet: "
                    + ", ".join(missing)
                ),
            )
        )
        return []

    try:
        results = search_azure_ai_search(query, records, top_k=top_k)
    except RuntimeError as error:
        steps.append(
            AgentStep(
                name="Azure AI Search",
                status="deferred",
                detail=str(error),
            )
        )
        return []

    steps.append(
        AgentStep(
            name="Azure AI Search",
            status="done",
            detail=(
                f"Asked Azure AI Search for optional candidates and matched "
                f"{len(results)} returned documents to the local metadata index."
            ),
        )
    )
    return results


def merge_search_results(
    left: list[SearchResult],
    right: list[SearchResult],
) -> list[SearchResult]:
    merged: dict[str, SearchResult] = {}

    for result in left + right:
        file_id = str(result.record.get("file_id"))
        current = merged.get(file_id)
        if current is None:
            merged[file_id] = SearchResult(
                record=result.record,
                score=result.score,
                reasons=list(result.reasons),
            )
            continue

        current.score += result.score
        current.reasons = dedupe(current.reasons + result.reasons)

    return sorted(merged.values(), key=lambda result: result.score, reverse=True)


def normalize_result_scores(results: list[SearchResult]) -> None:
    if not results:
        return

    max_score = max(result.score for result in results)
    if max_score <= 0:
        for result in results:
            result.score = 0.0
        return

    for result in results:
        result.score = round(max(0.0, min(100.0, result.score / max_score * 100)), 1)


def format_understanding_detail(understanding: QueryUnderstanding) -> str:
    parts: list[str] = []

    if understanding.metadata_clues:
        parts.append("metadata clues: " + ", ".join(understanding.metadata_clues))
    if understanding.content_clues:
        parts.append("content clues: " + ", ".join(understanding.content_clues))
    if understanding.visual_clues:
        parts.append("visual clues: " + ", ".join(understanding.visual_clues))

    if not parts:
        return "No strong clue type detected, so start with metadata search."

    return "; ".join(parts)


def should_run_content_inspection(
    understanding: QueryUnderstanding,
    results: list[SearchResult],
    content_mode: str,
) -> bool:
    if content_mode == "always":
        return True
    if content_mode == "never":
        return False
    if understanding.should_inspect_content or understanding.should_inspect_visuals:
        return True
    if not results:
        return True
    return results[0].score < 8


def should_run_visual_inspection(
    understanding: QueryUnderstanding,
    results: list[SearchResult],
    visual_mode: str,
) -> bool:
    if visual_mode == "azure":
        return bool(results)
    if visual_mode == "never":
        return False
    return bool(results and understanding.should_inspect_visuals)


def should_run_clip_prefilter(
    understanding: QueryUnderstanding,
    results: list[SearchResult],
    visual_prefilter: str,
) -> bool:
    if visual_prefilter == "none":
        return False
    return bool(results and understanding.should_inspect_visuals)


def make_content_skipped_step(
    understanding: QueryUnderstanding,
    results: list[SearchResult],
    content_mode: str,
) -> AgentStep:
    if not results:
        return AgentStep(
            name="Inspect Content",
            status="skipped",
            detail="No candidate files were found from metadata.",
        )

    if content_mode == "never":
        return AgentStep(
            name="Inspect Content",
            status="skipped",
            detail="Content inspection was disabled for this run.",
        )

    return AgentStep(
        name="Inspect Content",
        status="not needed yet",
        detail=(
            "The query appears answerable from filename, folder, file type, and "
            "basic metadata for this MVP step."
        ),
    )


def make_content_step(content_evidence: list[ContentEvidence]) -> AgentStep:
    inspected_files = len(content_evidence)
    inspected_pages = sum(evidence.inspected_pages for evidence in content_evidence)
    matched_files = sum(1 for evidence in content_evidence if evidence.score > 0)
    cache_hits = sum(1 for evidence in content_evidence if evidence.cache_hit)
    ocr_pages = sum(evidence.ocr_pages for evidence in content_evidence)

    errors = [evidence.error for evidence in content_evidence if evidence.error]
    ocr_errors = [
        evidence.ocr_error
        for evidence in content_evidence
        if evidence.ocr_error
    ]
    ocr_detail = (
        f" Local OCR fallback contributed text from {ocr_pages} pages."
        if ocr_pages
        else ""
    )
    if ocr_errors:
        ocr_detail += " OCR fallback notes: " + "; ".join(ocr_errors[:2]) + "."
    if errors:
        return AgentStep(
            name="Inspect Content",
            status="partial",
            detail=(
                f"Read text from {inspected_files} candidate files. "
                f"{matched_files} files had matching text. "
                f"Some files were skipped: {'; '.join(errors)}."
                f"{ocr_detail}"
            ),
        )

    return AgentStep(
        name="Inspect Content",
        status="done",
        detail=(
            f"Read text from {inspected_files} candidate files across "
            f"{inspected_pages} pages/slides/sheets. {matched_files} files had "
            f"matching text.{ocr_detail}"
        ),
    )


def should_run_content_llm_rerank(
    mode: str,
    content_evidence: list[ContentEvidence],
) -> bool:
    if mode != "llm":
        return False
    return any(
        not evidence.error and (evidence.text_samples or evidence.snippets)
        for evidence in content_evidence
    )


def make_content_llm_step(results: list[SearchResult]) -> AgentStep:
    return AgentStep(
        name="Content LLM Rerank",
        status="done",
        detail=(
            "Used the deep Azure OpenAI deployment to compare extracted text "
            f"evidence and return the top {len(results)} matches."
        ),
    )


def make_content_llm_deferred_step(error: str) -> AgentStep:
    return AgentStep(
        name="Content LLM Rerank",
        status="deferred",
        detail=(
            "Deep content reranking could not run, so the agent kept the local "
            f"text ranking. {error}"
        ),
    )


def make_clip_step(
    clip_evidence: list[ClipEvidence],
    max_clip_pages: int,
) -> AgentStep:
    rendered_files = len(clip_evidence)
    rendered_pages = sum(evidence.rendered_pages for evidence in clip_evidence)
    selected_pages = sum(len(evidence.selected_pages) for evidence in clip_evidence)
    cache_hits = sum(evidence.cache_hits for evidence in clip_evidence)
    errors = [evidence.error for evidence in clip_evidence if evidence.error]

    if errors:
        return AgentStep(
            name="CLIP Visual Prefilter",
            status="partial",
            detail=(
                f"Selected likely visual pages with CLIP before Azure Vision. "
                f"Some files were skipped: {'; '.join(errors)}."
            ),
        )

    return AgentStep(
        name="CLIP Visual Prefilter",
        status="done",
        detail=(
            f"Selected {selected_pages} likely visual pages from "
            f"{rendered_files} candidate files before Azure Vision."
        ),
    )


def make_clip_error_step(error: str) -> AgentStep:
    if "GPU CLIP worker" in error or "CLIP_WORKER_URL" in error:
        detail = (
            "The GPU CLIP worker could not select visual pages, so visual "
            f"inspection is waiting for the worker setup. {error}"
        )
    else:
        detail = (
            "CLIP is not installed in this runtime, so local visual page "
            "selection is waiting for the optional visual worker/dependency setup."
        )

    return AgentStep(
        name="CLIP Visual Prefilter",
        status="deferred",
        detail=detail,
    )


def make_clip_skipped_step(visual_prefilter: str) -> AgentStep:
    status = "skipped" if visual_prefilter == "none" else "not needed yet"
    detail = (
        "CLIP visual prefilter was disabled for this run."
        if visual_prefilter == "none"
        else "No visual clues were detected, so CLIP prefilter was not needed."
    )
    return AgentStep(name="CLIP Visual Prefilter", status=status, detail=detail)


def make_visual_step(visual_evidence: list[VisualEvidence]) -> AgentStep:
    analyzed_files = len(visual_evidence)
    analyzed_pages = sum(evidence.analyzed_pages for evidence in visual_evidence)
    matched_files = sum(1 for evidence in visual_evidence if evidence.score > 0)
    azure_calls = sum(evidence.azure_calls for evidence in visual_evidence)
    cache_hits = sum(evidence.cache_hits for evidence in visual_evidence)
    errors = [evidence.error for evidence in visual_evidence if evidence.error]

    if errors:
        return AgentStep(
            name="Inspect Visuals",
            status="partial",
            detail=(
                f"Checked selected pages with Azure Vision. "
                f"{matched_files} files had visual matches. "
                f"Some files were skipped: {'; '.join(errors)}."
            ),
        )

    return AgentStep(
        name="Inspect Visuals",
        status="done",
        detail=(
            f"Checked selected pages from {analyzed_files} files with Azure Vision. "
            f"{matched_files} files had visual matches."
        ),
    )


def make_visual_blocked_by_clip_step() -> AgentStep:
    return AgentStep(
        name="Inspect Visuals",
        status="skipped",
        detail=(
            "Azure Vision was skipped because CLIP page selection was unavailable. "
            "This avoids billable analysis of broad, unfiltered pages."
        ),
    )


def make_visual_missing_config_step(missing_config: list[str]) -> AgentStep:
    return AgentStep(
        name="Inspect Visuals",
        status="deferred",
        detail=(
            "Azure Vision is selected for visual inspection, but credentials are "
            "not configured yet. Add "
            + ", ".join(missing_config)
            + " to .env before running billable visual analysis."
        ),
    )


def make_visual_skipped_step(visual_mode: str) -> AgentStep:
    status = "skipped" if visual_mode == "never" else "deferred"
    detail = (
        "Visual inspection was disabled for this run."
        if visual_mode == "never"
        else "Visual inspection is available, but was not selected for this run."
    )
    return AgentStep(name="Inspect Visuals", status=status, detail=detail)


def apply_content_evidence(
    results: list[SearchResult],
    content_evidence: list[ContentEvidence],
) -> None:
    evidence_by_file_id = {
        evidence.file_id: evidence
        for evidence in content_evidence
    }

    for result in results:
        file_id = str(result.record.get("file_id"))
        evidence = evidence_by_file_id.get(file_id)
        if not evidence:
            continue

        if evidence.error:
            result.reasons.append(evidence.error)
            continue

        if evidence.score <= 0:
            if evidence.ocr_used:
                result.reasons.append(
                    f"local OCR fallback added text from {evidence.ocr_pages} pages"
                )
            elif evidence.ocr_error:
                result.reasons.append("local OCR fallback note: " + evidence.ocr_error)
            continue

        meaningful_terms = evidence.matched_terms
        if meaningful_terms:
            result.score += evidence.score
            result.reasons.append(
                "content text matched: " + ", ".join(meaningful_terms)
            )
        if evidence.ocr_used:
            result.reasons.append(
                f"local OCR fallback added text from {evidence.ocr_pages} pages"
            )
        elif evidence.ocr_error:
            result.reasons.append("local OCR fallback note: " + evidence.ocr_error)
        for snippet in filter_snippets_for_terms(evidence.snippets, meaningful_terms):
            result.reasons.append("content snippet " + snippet)


def filter_snippets_for_terms(snippets: list[str], terms: list[str]) -> list[str]:
    if not terms:
        return []

    normalized_terms = [term.lower() for term in terms]
    return [
        snippet
        for snippet in snippets
        if any(term in snippet.lower() for term in normalized_terms)
    ]


def apply_clip_evidence(
    results: list[SearchResult],
    clip_evidence: list[ClipEvidence],
) -> None:
    evidence_by_file_id = {
        evidence.file_id: evidence
        for evidence in clip_evidence
    }

    for result in results:
        file_id = str(result.record.get("file_id"))
        evidence = evidence_by_file_id.get(file_id)
        if not evidence:
            continue

        if evidence.error:
            continue

        if evidence.score <= 0:
            continue

        result.score += evidence.score
        result.reasons.append(
            "Visual page candidates: "
            + ", ".join(str(page) for page in evidence.selected_pages)
        )


def apply_visual_evidence(
    results: list[SearchResult],
    visual_evidence: list[VisualEvidence],
) -> None:
    evidence_by_file_id = {
        evidence.file_id: evidence
        for evidence in visual_evidence
    }

    for result in results:
        file_id = str(result.record.get("file_id"))
        evidence = evidence_by_file_id.get(file_id)
        if not evidence:
            continue

        if evidence.error:
            result.reasons.append(evidence.error)
            continue

        if evidence.score <= 0:
            continue

        result.score += evidence.score
        result.reasons.append(
            "Visual analysis matched: " + ", ".join(evidence.matched_terms)
        )
        for observation in evidence.observations[:2]:
            result.reasons.append("visual observation " + observation)


def parse_content_cache_path(value: str) -> Path:
    return Path(value)


def parse_cache_path(value: str) -> Path:
    return Path(value)


def parse_positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def make_answer_step(results: list[SearchResult]) -> AgentStep:
    if not results:
        return AgentStep(
            name="Return Answer",
            status="done",
            detail="No match found from metadata.",
        )

    best = results[0].record
    return AgentStep(
        name="Return Answer",
        status="done",
        detail=f"Best current match is {best.get('filename')}.",
    )


def print_response(response: AgentResponse) -> None:
    print(f"Query: {response.query_understanding.query}")
    print(f"Evidence used: {response.evidence_scope}")
    print()
    print("Agent Steps")
    for index, step in enumerate(response.steps, start=1):
        print(f"{index}. {step.name} [{step.status}]")
        print(f"   {step.detail}")
    print()

    if not response.results:
        print("No metadata matches found.")
        return

    print("Results")
    for index, result in enumerate(response.results, start=1):
        record = result.record
        print(f"{index}. {record.get('filename')}")
        print(f"   score: {result.score:.2f}")
        print(f"   path: {record.get('relative_path')}")
        print(f"   type: {record.get('type_label')} | size: {record.get('file_size_label', 'unknown')}")
        print("   reasons:")
        for reason in result.reasons:
            print(f"   - {reason}")
        print()


def response_to_json(response: AgentResponse) -> dict[str, Any]:
    understanding = response.query_understanding
    return {
        "query": understanding.query,
        "evidence_scope": response.evidence_scope,
        "query_understanding": {
            "metadata_clues": understanding.metadata_clues,
            "content_clues": understanding.content_clues,
            "visual_clues": understanding.visual_clues,
            "should_inspect_content": understanding.should_inspect_content,
            "should_inspect_visuals": understanding.should_inspect_visuals,
        },
        "steps": [
            {
                "name": step.name,
                "status": step.status,
                "detail": step.detail,
            }
            for step in response.steps
        ],
        "results": serialize_results(response.results),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the document search agent.")
    parser.add_argument("--query", required=True, help="Natural-language search query.")
    parser.add_argument(
        "--index",
        type=Path,
        default=DEFAULT_INDEX_PATH,
        help="Path to files_index.json.",
    )
    parser.add_argument(
        "--mode",
        choices=["local", "llm"],
        default="local",
        help="Use local metadata search or Azure OpenAI metadata reranking.",
    )
    parser.add_argument(
        "--orchestration-mode",
        choices=["manual", "semantic-kernel"],
        default="manual",
        help="Use fixed settings or Semantic Kernel auto-mode planning.",
    )
    parser.add_argument(
        "--azure-ai-search-mode",
        choices=["auto", "never"],
        default="never",
        help="Use Azure AI Search as an optional candidate retrieval tool.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Number of results.")
    parser.add_argument(
        "--candidate-pool-size",
        type=int,
        default=8,
        help="How many local candidates to send to Azure OpenAI in llm mode.",
    )
    parser.add_argument(
        "--content-mode",
        choices=["auto", "always", "never"],
        default="auto",
        help="When to inspect extracted text from candidate files.",
    )
    parser.add_argument(
        "--translator-mode",
        choices=["auto", "never"],
        default="never",
        help="Use Azure Translator to expand Japanese queries into English.",
    )
    parser.add_argument(
        "--max-inspected-files",
        type=parse_positive_int,
        default=6,
        help="Maximum number of candidate files to open for content inspection.",
    )
    parser.add_argument(
        "--max-pages-per-file",
        type=parse_positive_int,
        default=50,
        help="Maximum pages/slides/chunks/sheets to extract per inspected file.",
    )
    parser.add_argument(
        "--max-chars-per-file",
        type=parse_positive_int,
        default=30000,
        help="Maximum extracted characters per inspected file.",
    )
    parser.add_argument(
        "--content-cache",
        type=parse_content_cache_path,
        default=DEFAULT_CONTENT_CACHE_PATH,
        help="Path to the generated content text cache.",
    )
    parser.add_argument(
        "--visual-mode",
        choices=["auto", "azure", "never"],
        default="never",
        help="When to analyze rendered candidate pages with Azure Vision.",
    )
    parser.add_argument(
        "--visual-prefilter",
        choices=["clip", "none"],
        default="none",
        help="Use local CLIP to select visual pages before Azure Vision.",
    )
    parser.add_argument(
        "--max-visual-files",
        type=parse_positive_int,
        default=3,
        help="Maximum number of candidate files to send to Azure Vision.",
    )
    parser.add_argument(
        "--max-visual-pages-per-file",
        type=parse_positive_int,
        default=5,
        help="Maximum rendered pages per file to send to Azure Vision.",
    )
    parser.add_argument(
        "--max-clip-pages",
        type=parse_positive_int,
        default=10,
        help="Maximum CLIP-selected pages to send to Azure Vision.",
    )
    parser.add_argument(
        "--visual-cache",
        type=parse_cache_path,
        default=DEFAULT_VISUAL_CACHE_PATH,
        help="Path to the generated Azure Vision analysis cache.",
    )
    parser.add_argument(
        "--clip-cache",
        type=parse_cache_path,
        default=DEFAULT_CLIP_CACHE_PATH,
        help="Path to the generated CLIP embedding cache.",
    )
    parser.add_argument(
        "--search-memory-cache",
        type=parse_cache_path,
        default=DEFAULT_SEARCH_MEMORY_PATH,
        help="Path to the generated search memory cache.",
    )
    parser.add_argument(
        "--extension",
        action="append",
        dest="extensions",
        help="Limit search to an extension. Repeat for multiple extensions, e.g. --extension .pdf --extension .pptx.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON output.",
    )

    args = parser.parse_args()
    load_environment()

    response = run_agent(
        query=args.query,
        index_path=args.index,
        mode=args.mode,
        top_k=args.top_k,
        candidate_pool_size=args.candidate_pool_size,
        content_mode=args.content_mode,
        translator_mode=args.translator_mode,
        max_inspected_files=args.max_inspected_files,
        max_pages_per_file=args.max_pages_per_file,
        max_chars_per_file=args.max_chars_per_file,
        content_cache_path=args.content_cache,
        visual_mode=args.visual_mode,
        visual_prefilter=args.visual_prefilter,
        max_visual_files=args.max_visual_files,
        max_visual_pages_per_file=args.max_visual_pages_per_file,
        max_clip_pages=args.max_clip_pages,
        visual_cache_path=args.visual_cache,
        clip_cache_path=args.clip_cache,
        orchestration_mode=args.orchestration_mode,
        azure_ai_search_mode=args.azure_ai_search_mode,
        search_memory_path=args.search_memory_cache,
        allowed_extensions=set(args.extensions) if args.extensions else None,
    )

    if args.json:
        print(json.dumps(response_to_json(response), ensure_ascii=False, indent=2))
        return

    print_response(response)


if __name__ == "__main__":
    main()
