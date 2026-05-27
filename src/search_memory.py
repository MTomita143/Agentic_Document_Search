"""Lightweight memory of previous search sessions.

This is not a document index. It remembers what the agent already investigated,
so future vague queries can start from recently useful files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from search import SearchResult, dedupe, normalize_text, tokenize


DEFAULT_SEARCH_MEMORY_PATH = Path("indexes/search_memory.json")
MAX_MEMORY_EVENTS = 200
MEMORY_QUERY_TERMS = {
    "before",
    "earlier",
    "last",
    "month",
    "previous",
    "recent",
    "recently",
    "searched",
    "探した",
    "前",
    "以前",
    "先月",
    "最近",
    "検索",
}


@dataclass
class MemoryEvidence:
    file_id: str
    score: float
    reasons: list[str]


def load_memory(cache_path: Path = DEFAULT_SEARCH_MEMORY_PATH) -> dict[str, Any]:
    if not cache_path.exists():
        return {"events": []}

    with cache_path.open("r", encoding="utf-8") as f:
        loaded = json.load(f)

    if isinstance(loaded, dict) and isinstance(loaded.get("events"), list):
        return loaded

    return {"events": []}


def save_memory(
    memory: dict[str, Any],
    cache_path: Path = DEFAULT_SEARCH_MEMORY_PATH,
) -> None:
    events = memory.get("events", [])
    if isinstance(events, list) and len(events) > MAX_MEMORY_EVENTS:
        memory["events"] = events[-MAX_MEMORY_EVENTS:]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)


def find_memory_candidates(
    query: str,
    records: list[dict[str, Any]],
    top_k: int,
    cache_path: Path = DEFAULT_SEARCH_MEMORY_PATH,
) -> list[SearchResult]:
    memory = load_memory(cache_path)
    events = list(memory.get("events", []))
    if not events:
        return []

    records_by_file_id = {
        str(record.get("file_id")): record
        for record in records
    }
    scores: dict[str, float] = {}
    reasons_by_file_id: dict[str, list[str]] = {}
    memory_query = is_memory_query(query)

    for age_index, event in enumerate(reversed(events)):
        similarity = query_similarity(query, str(event.get("query", "")))
        if similarity <= 0 and not memory_query:
            continue

        recency_bonus = max(0.0, 3.0 - age_index * 0.1)
        event_score = similarity * 16.0 + (recency_bonus if memory_query else 0.0)
        if event_score <= 0:
            continue

        for rank, result in enumerate(event.get("results", [])[:3], start=1):
            file_id = str(result.get("file_id", ""))
            if not file_id or file_id not in records_by_file_id:
                continue

            rank_bonus = max(0.0, 4.0 - rank)
            scores[file_id] = scores.get(file_id, 0.0) + event_score + rank_bonus
            reasons_by_file_id.setdefault(file_id, []).append(
                f"search memory: matched a previous search '{event.get('query')}'"
            )

    results = [
        SearchResult(
            record=records_by_file_id[file_id],
            score=score,
            reasons=dedupe(reasons_by_file_id.get(file_id, []))[:2],
        )
        for file_id, score in scores.items()
    ]
    return sorted(results, key=lambda result: result.score, reverse=True)[:top_k]


def apply_memory_evidence(
    query: str,
    results: list[SearchResult],
    cache_path: Path = DEFAULT_SEARCH_MEMORY_PATH,
) -> list[MemoryEvidence]:
    memory = load_memory(cache_path)
    events = list(memory.get("events", []))
    if not events or not results:
        return []

    file_ids = {str(result.record.get("file_id")) for result in results}
    evidence_by_file_id: dict[str, MemoryEvidence] = {}

    for event in reversed(events[-50:]):
        similarity = query_similarity(query, str(event.get("query", "")))
        if similarity <= 0:
            continue

        for prior_result in event.get("results", [])[:3]:
            file_id = str(prior_result.get("file_id", ""))
            if file_id not in file_ids:
                continue

            current = evidence_by_file_id.get(file_id)
            reason = f"previously appeared in a related search: {event.get('query')}"
            score = similarity * 8.0
            if current is None:
                evidence_by_file_id[file_id] = MemoryEvidence(
                    file_id=file_id,
                    score=score,
                    reasons=[reason],
                )
            else:
                current.score += score
                current.reasons.append(reason)

    evidence = list(evidence_by_file_id.values())
    evidence_by_file_id = {item.file_id: item for item in evidence}
    for result in results:
        file_id = str(result.record.get("file_id"))
        item = evidence_by_file_id.get(file_id)
        if not item:
            continue
        result.score += item.score
        result.reasons.extend(dedupe(item.reasons))

    return evidence


def remember_search(
    query: str,
    results: list[SearchResult],
    cache_path: Path = DEFAULT_SEARCH_MEMORY_PATH,
) -> None:
    if not results:
        return

    memory = load_memory(cache_path)
    events = memory.setdefault("events", [])
    events.append(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "query_terms": tokenize(query),
            "results": [
                {
                    "file_id": result.record.get("file_id"),
                    "uri": result.record.get("uri") or result.record.get("relative_path"),
                    "filename": result.record.get("filename"),
                    "score": result.score,
                    "reasons": result.reasons[:3],
                }
                for result in results[:5]
            ],
        }
    )
    save_memory(memory, cache_path)


def query_similarity(left: str, right: str) -> float:
    left_terms = set(tokenize(left))
    right_terms = set(tokenize(right))
    left_terms -= MEMORY_QUERY_TERMS
    right_terms -= MEMORY_QUERY_TERMS

    if not left_terms or not right_terms:
        return 0.0

    overlap = left_terms & right_terms
    return len(overlap) / max(len(left_terms), len(right_terms))


def is_memory_query(query: str) -> bool:
    normalized = normalize_text(query)
    query_terms = set(tokenize(query))
    return bool(query_terms & MEMORY_QUERY_TERMS) or any(
        term in normalized
        for term in MEMORY_QUERY_TERMS
        if len(term) > 1
    )
