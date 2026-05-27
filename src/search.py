"""
Search the file-level metadata index.

MVP v0 deliberately searches metadata only. It does not open PDF, PPTX, or DOCX
contents. The LLM mode can reason over candidate metadata, but it receives no
document text or visual information.

Usage:
    python3 src/search.py --query "investor presentation" --mode local --top-k 3
    python3 src/search.py --query "deck for the board" --mode llm --top-k 3
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from metadata import hydrate_records


DEFAULT_INDEX_PATH = Path("indexes/files_index.json")
DEFAULT_AZURE_OPENAI_API_VERSION = "2024-12-01-preview"
CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uff66-\uff9f]+")
TOKEN_RE = re.compile(r"[a-z0-9]+|[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uff66-\uff9f]+")
SINGLE_CJK_TERMS = {"表", "図", "青", "赤", "黒", "白"}
JAPANESE_DOMAIN_TERMS = {
    "売上",
    "予算",
    "前年比",
    "投資",
    "投資家",
    "市場",
    "分析",
    "調査",
    "報告",
    "報告書",
    "資料",
    "戦略",
    "収益",
    "財務",
    "決算",
    "会議",
    "計画",
    "製品",
    "開発",
    "研究",
    "通信",
    "金融",
}
ENGLISH_TOKEN_ALIASES = {
    "charts": ["chart"],
    "decks": ["deck"],
    "documents": ["document"],
    "graphs": ["graph"],
    "images": ["image"],
    "pictures": ["picture"],
    "presentations": ["presentation"],
    "reports": ["report"],
    "slides": ["slide"],
    "tables": ["table"],
}
NEGATIVE_REASON_MARKERS = {
    "do not",
    "does not",
    "doesn't",
    "don't",
    "lack",
    "lacks",
    "missing",
    "no ",
    "not ",
    "without",
}

FIELD_WEIGHTS = {
    "filename": 5,
    "title": 5,
    "folder_path": 3,
    "parent_folder": 3,
    "grandparent_folder": 2,
    "type_label": 2,
    "extension": 1,
    "file_size_label": 1,
}

STOPWORDS = {
    "a",
    "able",
    "about",
    "also",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "by",
    "can",
    "company",
    "could",
    "did",
    "do",
    "does",
    "drive",
    "find",
    "for",
    "from",
    "get",
    "had",
    "has",
    "have",
    "i",
    "in",
    "is",
    "it",
    "like",
    "looking",
    "me",
    "may",
    "might",
    "of",
    "on",
    "one",
    "or",
    "probably",
    "remember",
    "should",
    "some",
    "something",
    "that",
    "the",
    "these",
    "this",
    "those",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "will",
    "with",
    "would",
}


@dataclass
class SearchResult:
    record: dict[str, Any]
    score: float
    reasons: list[str]


def load_index(index_path: Path) -> list[dict[str, Any]]:
    if not index_path.exists():
        raise FileNotFoundError(
            f"Index not found: {index_path}. Run src/ingest.py first."
        )

    with index_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        raise ValueError(f"Expected a list of records in {index_path}")

    return hydrate_records(records, index_path=index_path)


def load_environment() -> None:
    """Load local .env values when python-dotenv is installed."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    load_dotenv()


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"[_\-/\\.,:;()\[\]{}、。・：；（）「」『』【】]+", " ", text)
    return text.lower()


def tokenize(value: Any) -> list[str]:
    normalized = normalize_text(value)
    tokens: list[str] = []

    for match in TOKEN_RE.finditer(normalized):
        chunk = match.group(0)
        if has_cjk(chunk):
            tokens.extend(tokenize_cjk_chunk(chunk))
        else:
            tokens.append(chunk)
            tokens.extend(ENGLISH_TOKEN_ALIASES.get(chunk, []))

    return dedupe_tokens(
        token
        for token in tokens
        if token and token not in STOPWORDS
    )


def tokenize_cjk_chunk(chunk: str) -> list[str]:
    tokens: list[str] = []

    for run, script in iter_cjk_script_runs(chunk):
        if script == "hiragana":
            continue
        tokens.extend(tokenize_cjk_noun_run(run, script))

    return tokens


def iter_cjk_script_runs(chunk: str) -> list[tuple[str, str]]:
    runs: list[tuple[str, str]] = []
    current = ""
    current_script = ""

    for char in chunk:
        script = cjk_script(char)
        if current and script != current_script:
            runs.append((current, current_script))
            current = ""
        current += char
        current_script = script

    if current:
        runs.append((current, current_script))

    return runs


def cjk_script(char: str) -> str:
    codepoint = ord(char)
    if 0x3040 <= codepoint <= 0x309F:
        return "hiragana"
    if 0x30A0 <= codepoint <= 0x30FF or 0xFF66 <= codepoint <= 0xFF9F:
        return "katakana"
    return "kanji"


def tokenize_cjk_noun_run(chunk: str, script: str) -> list[str]:
    if len(chunk) <= 1:
        return [chunk] if chunk in SINGLE_CJK_TERMS else []

    tokens = [chunk]
    tokens.extend(
        term
        for term in JAPANESE_DOMAIN_TERMS
        if term != chunk and term in chunk
    )

    return tokens


def has_cjk(value: str) -> bool:
    return CJK_RE.search(value) is not None


def dedupe_tokens(values: Any) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []

    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)

    return output


def build_search_text(record: dict[str, Any]) -> dict[str, str]:
    return {
        field: normalize_text(record.get(field, ""))
        for field in FIELD_WEIGHTS
    }


def score_record(query: str, record: dict[str, Any]) -> SearchResult:
    query_tokens = tokenize(query)
    field_text = build_search_text(record)
    reasons: list[str] = []
    score = 0.0

    for field, text in field_text.items():
        if not text:
            continue

        field_tokens = tokenize(text)
        field_weight = FIELD_WEIGHTS[field]

        for query_token in query_tokens:
            if query_token in field_tokens:
                score += field_weight
                reasons.append(f"matched '{query_token}' in {field}")
                continue

            fuzzy_token = best_fuzzy_token(query_token, field_tokens)
            if fuzzy_token:
                score += field_weight * 0.65
                reasons.append(
                    f"fuzzy matched '{query_token}' to '{fuzzy_token}' in {field}"
                )

    exact_phrase = normalize_text(query).strip()
    if exact_phrase:
        for field in ("filename", "title", "folder_path"):
            if exact_phrase in field_text[field]:
                score += FIELD_WEIGHTS[field] * 2
                reasons.append(f"matched phrase in {field}")

    if mentions_slides(query) and looks_like_slides(record):
        score += 2
        reasons.append("query sounds like slides/deck and metadata suggests presentation")

    if mentions_report(query) and looks_like_report(record):
        score += 2
        reasons.append("query sounds like a report and metadata suggests report")

    return SearchResult(record=record, score=score, reasons=dedupe(reasons))


def best_fuzzy_token(query_token: str, field_tokens: list[str]) -> str | None:
    if has_cjk(query_token):
        return None

    if len(query_token) < 4:
        return None

    best_token = None
    best_ratio = 0.0

    for field_token in field_tokens:
        if len(field_token) < 4:
            continue
        ratio = SequenceMatcher(None, query_token, field_token).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_token = field_token

    if best_ratio >= 0.82:
        return best_token
    return None


def mentions_slides(query: str) -> bool:
    normalized = normalize_text(query)
    if any(word in normalized for word in ("スライド", "プレゼン", "発表資料")):
        return True

    query_tokens = set(tokenize(query))
    return bool(query_tokens & {"deck", "presentation", "slide", "slides"})


def mentions_report(query: str) -> bool:
    normalized = normalize_text(query)
    if any(word in normalized for word in ("レポート", "報告", "報告書", "調査")):
        return True

    query_tokens = set(tokenize(query))
    return bool(query_tokens & {"annual", "report", "research"})


def looks_like_slides(record: dict[str, Any]) -> bool:
    text = normalize_text(
        " ".join(
            str(record.get(field, ""))
            for field in ("filename", "title", "folder_path", "type_label")
        )
    )
    return any(
        word in text
        for word in (
            "deck",
            "presentation",
            "slide",
            "slides",
            "スライド",
            "プレゼン",
            "発表資料",
        )
    )


def looks_like_report(record: dict[str, Any]) -> bool:
    text = normalize_text(
        " ".join(
            str(record.get(field, ""))
            for field in ("filename", "title", "folder_path", "type_label")
        )
    )
    return any(
        word in text
        for word in (
            "annual",
            "report",
            "research",
            "レポート",
            "報告",
            "報告書",
            "調査",
        )
    )


def dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []

    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)

    return output


def search_local(
    query: str,
    records: list[dict[str, Any]],
    top_k: int,
    include_zero_scores: bool = False,
) -> list[SearchResult]:
    results = [score_record(query, record) for record in records]
    if not include_zero_scores:
        results = [result for result in results if result.score > 0]

    return sorted(
        results,
        key=lambda result: (
            result.score,
            -len(str(result.record.get("relative_path", ""))),
        ),
        reverse=True,
    )[:top_k]


def search_llm(
    query: str,
    records: list[dict[str, Any]],
    top_k: int,
    candidate_pool_size: int,
) -> list[SearchResult]:
    candidates = search_local(
        query,
        records,
        top_k=min(candidate_pool_size, len(records)),
        include_zero_scores=True,
    )

    if not candidates:
        return []

    llm_rankings = rerank_with_azure_openai(query, candidates, top_k)
    by_file_id = {
        candidate.record["file_id"]: candidate
        for candidate in candidates
    }

    results: list[SearchResult] = []
    for ranking in llm_rankings:
        file_id = ranking.get("file_id")
        candidate = by_file_id.get(file_id)
        if not candidate:
            continue

        score = float(ranking.get("score", candidate.score))
        reasons = ranking.get("reasons", [])
        if not isinstance(reasons, list):
            reasons = [str(reasons)]
        positive_reasons = sanitize_reasons([str(reason) for reason in reasons])
        if not positive_reasons:
            positive_reasons = candidate.reasons
        if not positive_reasons and candidate.score <= 0:
            continue

        results.append(
            SearchResult(
                record=candidate.record,
                score=score,
                reasons=positive_reasons,
            )
        )

    return results[:top_k]


def sanitize_reasons(reasons: list[str]) -> list[str]:
    return [
        reason
        for reason in dedupe(reasons)
        if reason.strip() and not is_negative_reason(reason)
    ]


def is_negative_reason(reason: str) -> bool:
    normalized = normalize_text(reason)
    return any(marker in normalized for marker in NEGATIVE_REASON_MARKERS)


def rerank_with_azure_openai(
    query: str,
    candidates: list[SearchResult],
    top_k: int,
) -> list[dict[str, Any]]:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    api_version = os.getenv(
        "AZURE_OPENAI_API_VERSION",
        DEFAULT_AZURE_OPENAI_API_VERSION,
    )

    missing = [
        name
        for name, value in {
            "AZURE_OPENAI_ENDPOINT": endpoint,
            "AZURE_OPENAI_API_KEY": api_key,
            "AZURE_OPENAI_DEPLOYMENT": deployment,
        }.items()
        if not value
    ]
    if missing:
        raise RuntimeError(
            "LLM mode needs Azure OpenAI environment variables: "
            + ", ".join(missing)
        )

    try:
        from openai import AzureOpenAI
    except ImportError as error:
        raise RuntimeError(
            "LLM mode needs the OpenAI SDK. Run: pip3 install -r requirements.txt"
        ) from error

    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=api_key,
    )
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a document search reranker. Rank files using only "
                    "the provided metadata. Do not claim to have read document "
                    "contents or seen visuals. Return strict JSON only."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "query": query,
                        "top_k": top_k,
                        "candidate_metadata": [
                            {
                                "file_id": candidate.record.get("file_id"),
                                "filename": candidate.record.get("filename"),
                                "title": candidate.record.get("title"),
                                "extension": candidate.record.get("extension"),
                                "relative_path": candidate.record.get("relative_path"),
                                "folder_path": candidate.record.get("folder_path"),
                                "parent_folder": candidate.record.get("parent_folder"),
                                "grandparent_folder": candidate.record.get(
                                    "grandparent_folder"
                                ),
                                "type_label": candidate.record.get("type_label"),
                                "file_size_label": candidate.record.get(
                                    "file_size_label"
                                ),
                                "local_score": candidate.score,
                                "local_reasons": candidate.reasons,
                            }
                            for candidate in candidates
                        ],
                        "response_schema": {
                            "results": [
                                {
                                    "file_id": "string",
                                    "score": "number from 0 to 100",
                                    "reasons": [
                                        "short reason based only on metadata"
                                    ],
                                }
                            ]
                        },
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        max_completion_tokens=2000,
        model=deployment,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    if not content:
        raise ValueError("Azure OpenAI returned an empty message")

    parsed = json.loads(content)
    results = parsed.get("results", [])
    if not isinstance(results, list):
        raise ValueError("LLM response JSON must contain a results list")

    return results


def format_result(result: SearchResult, rank: int) -> str:
    record = result.record
    reason_lines = "\n".join(f"   - {reason}" for reason in result.reasons)
    if not reason_lines:
        reason_lines = "   - metadata was included in the candidate pool"

    return "\n".join(
        [
            f"{rank}. {record.get('filename')}",
            f"   score: {bounded_score(result.score):.2f}",
            f"   path: {record.get('relative_path')}",
            f"   type: {record.get('type_label')} | size: {record.get('file_size_label', 'unknown')}",
            "   reasons:",
            reason_lines,
        ]
    )


def bounded_score(score: float) -> float:
    return max(0.0, min(100.0, score))


def serialize_results(results: list[SearchResult]) -> list[dict[str, Any]]:
    return [
        {
            "file_id": result.record.get("file_id"),
            "filename": result.record.get("filename"),
            "relative_path": result.record.get("relative_path"),
            "type_label": result.record.get("type_label"),
            "file_size_label": result.record.get("file_size_label"),
            "score": bounded_score(result.score),
            "reasons": result.reasons,
        }
        for result in results
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Search document metadata.")
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
        help="Use local metadata scoring or Azure OpenAI metadata reranking.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Number of results.")
    parser.add_argument(
        "--candidate-pool-size",
        type=int,
        default=8,
        help="How many local candidates to send to the LLM in llm mode.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON results.",
    )

    args = parser.parse_args()
    load_environment()
    records = load_index(args.index)

    if args.mode == "local":
        results = search_local(args.query, records, args.top_k)
    else:
        results = search_llm(
            args.query,
            records,
            args.top_k,
            args.candidate_pool_size,
        )

    if args.json:
        print(json.dumps(serialize_results(results), ensure_ascii=False, indent=2))
        return

    if not results:
        print("No metadata matches found.")
        return

    print(f"Query: {args.query}")
    print(f"Mode: {args.mode}")
    print()
    for rank, result in enumerate(results, start=1):
        print(format_result(result, rank))
        print()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
