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
from typing import Any

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
from search import (
    DEFAULT_INDEX_PATH,
    SearchResult,
    load_environment,
    load_index,
    normalize_text,
    search_llm,
    search_local,
    serialize_results,
    tokenize,
)
from visual import (
    DEFAULT_VISUAL_CACHE_PATH,
    VisualEvidence,
    inspect_visuals_for_results,
    missing_vision_config,
)


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


def run_agent(
    query: str,
    index_path: Path,
    mode: str,
    top_k: int,
    candidate_pool_size: int,
    content_mode: str,
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
) -> AgentResponse:
    if mode == "llm" and candidate_pool_size < top_k:
        candidate_pool_size = top_k
    if (
        visual_prefilter == "clip"
        and max_clip_pages < max_visual_files * max_visual_pages_per_file
    ):
        max_clip_pages = max_visual_files * max_visual_pages_per_file

    query_understanding = understand_query(query)
    records = load_index(index_path)
    steps: list[AgentStep] = []

    steps.append(
        AgentStep(
            name="Understand Query",
            status="done",
            detail=format_understanding_detail(query_understanding),
        )
    )

    if mode == "local":
        results = search_local(query, records, top_k=top_k)
        evidence_scope = "metadata only"
        steps.append(
            AgentStep(
                name="Metadata Search",
                status="done",
                detail=(
                    f"Searched {len(records)} file metadata records with local "
                    f"keyword and fuzzy matching. Returned top {len(results)}."
                ),
            )
        )
    else:
        results = search_llm(
            query,
            records,
            top_k=top_k,
            candidate_pool_size=candidate_pool_size,
        )
        evidence_scope = "metadata only with Azure OpenAI reranking"
        steps.append(
            AgentStep(
                name="Metadata Search + LLM Rerank",
                status="done",
                detail=(
                    f"Used local metadata search to create a candidate pool, "
                    f"then reranked metadata only with Azure OpenAI. Returned "
                    f"top {len(results)}."
                ),
            )
        )

    if should_run_content_inspection(query_understanding, results, content_mode):
        if not results:
            results = search_local(
                query,
                records,
                top_k=max_inspected_files,
                include_zero_scores=True,
            )
        content_evidence = inspect_content_for_results(
            query=query,
            results=results,
            max_files=max_inspected_files,
            max_pages_per_file=max_pages_per_file,
            max_chars_per_file=max_chars_per_file,
            cache_path=content_cache_path,
        )
        apply_content_evidence(results, content_evidence)
        results = sorted(results, key=lambda result: result.score, reverse=True)[:top_k]
        evidence_scope = f"{evidence_scope} + extracted candidate text"
        steps.append(make_content_step(content_evidence))
    else:
        steps.append(make_content_skipped_step(query_understanding, results, content_mode))

    selected_visual_pages: dict[str, list[int]] | None = None
    clip_prefilter_failed = False

    if should_run_clip_prefilter(query_understanding, results, visual_prefilter):
        try:
            clip_evidence, selected_visual_pages = rank_visual_pages_with_clip(
                query=query,
                results=results,
                max_files=max_visual_files,
                max_pages_per_file=max_visual_pages_per_file,
                top_pages=max_clip_pages,
                cache_path=clip_cache_path,
            )
            apply_clip_evidence(results, clip_evidence)
            results = sorted(results, key=lambda result: result.score, reverse=True)[:top_k]
            evidence_scope = f"{evidence_scope} + local CLIP page prefilter"
            steps.append(make_clip_step(clip_evidence, max_clip_pages))
        except RuntimeError as error:
            clip_prefilter_failed = True
            steps.append(make_clip_error_step(str(error)))
    elif query_understanding.should_inspect_visuals:
        steps.append(make_clip_skipped_step(visual_prefilter))

    if should_run_visual_inspection(query_understanding, results, visual_mode):
        if clip_prefilter_failed and visual_prefilter == "clip":
            steps.append(make_visual_blocked_by_clip_step())
        else:
            missing_visual_config = missing_vision_config()
            if missing_visual_config:
                steps.append(make_visual_missing_config_step(missing_visual_config))
            else:
                visual_evidence = inspect_visuals_for_results(
                    query=query,
                    results=results,
                    max_files=max_visual_files,
                    max_pages_per_file=max_visual_pages_per_file,
                    cache_path=visual_cache_path,
                    pages_by_file_id=selected_visual_pages,
                )
                apply_visual_evidence(results, visual_evidence)
                results = sorted(results, key=lambda result: result.score, reverse=True)[:top_k]
                evidence_scope = f"{evidence_scope} + Azure Vision page analysis"
                steps.append(make_visual_step(visual_evidence))
    elif query_understanding.should_inspect_visuals:
        steps.append(make_visual_skipped_step(visual_mode))

    normalize_result_scores(results)
    steps.append(make_answer_step(results))

    return AgentResponse(
        query_understanding=query_understanding,
        steps=steps,
        results=results,
        evidence_scope=evidence_scope,
    )


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
                f"Inspected {inspected_files} candidate files and {inspected_pages} "
                f"content units. {matched_files} files had text matches. "
                f"{cache_hits} cache hits. Some files were skipped: {'; '.join(errors)}."
                f"{ocr_detail}"
            ),
        )

    return AgentStep(
        name="Inspect Content",
        status="done",
        detail=(
            f"Inspected {inspected_files} candidate files and {inspected_pages} "
            f"content units. {matched_files} files had text matches. "
            f"{cache_hits} cache hits.{ocr_detail}"
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
                f"Ranked rendered pages locally with CLIP. Selected "
                f"{selected_pages} of at most {max_clip_pages} pages for visual "
                f"verification. Rendered {rendered_pages} pages across "
                f"{rendered_files} files, with {cache_hits} embedding cache hits. "
                f"Some files were skipped: {'; '.join(errors)}."
            ),
        )

    return AgentStep(
        name="CLIP Visual Prefilter",
        status="done",
        detail=(
            f"Ranked rendered pages locally with CLIP. Selected {selected_pages} "
            f"of at most {max_clip_pages} pages for visual verification. "
            f"Rendered {rendered_pages} pages across {rendered_files} files, "
            f"with {cache_hits} embedding cache hits."
        ),
    )


def make_clip_error_step(error: str) -> AgentStep:
    return AgentStep(
        name="CLIP Visual Prefilter",
        status="deferred",
        detail=error,
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
                f"Analyzed {analyzed_files} files and {analyzed_pages} pages with "
                f"Azure Vision. {matched_files} files had visual matches. "
                f"{azure_calls} Azure calls, {cache_hits} cache hits. "
                f"Some files were skipped: {'; '.join(errors)}."
            ),
        )

    return AgentStep(
        name="Inspect Visuals",
        status="done",
        detail=(
            f"Analyzed {analyzed_files} files and {analyzed_pages} rendered pages "
            f"with Azure Vision. {matched_files} files had visual matches. "
            f"{azure_calls} Azure calls, {cache_hits} cache hits."
        ),
    )


def make_visual_blocked_by_clip_step() -> AgentStep:
    return AgentStep(
        name="Inspect Visuals",
        status="skipped",
        detail=(
            "Azure Vision was not run because CLIP prefilter was selected but "
            "could not run. This avoids sending unfiltered pages to a billable "
            "visual verifier."
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
            result.reasons.append(evidence.error)
            continue

        if evidence.score <= 0:
            result.reasons.append(
                f"CLIP ranked {evidence.rendered_pages} rendered pages; no selected page improved the match"
            )
            continue

        result.score += evidence.score
        result.reasons.append(
            "CLIP selected visual pages: "
            + ", ".join(str(page) for page in evidence.selected_pages)
        )
        for observation in evidence.observations:
            result.reasons.append("CLIP observation " + observation)


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
            result.reasons.append(
                f"analyzed {evidence.analyzed_pages} rendered pages with Azure Vision; no visual query terms matched"
            )
            continue

        result.score += evidence.score
        result.reasons.append(
            "visual analysis matched terms: " + ", ".join(evidence.matched_terms)
        )
        for observation in evidence.observations:
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
    )

    if args.json:
        print(json.dumps(response_to_json(response), ensure_ascii=False, indent=2))
        return

    print_response(response)


if __name__ == "__main__":
    main()
