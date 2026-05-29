"""Deep Azure OpenAI reranking over extracted candidate text."""

from __future__ import annotations

import json
from typing import Any

from azure_openai_config import (
    deployment_candidates,
    format_deployment_not_found_message,
    get_azure_openai_config,
    is_deployment_not_found_error,
)
from content import ContentEvidence
from search import SearchResult, sanitize_reasons


def rerank_with_content_azure_openai(
    query: str,
    results: list[SearchResult],
    content_evidence: list[ContentEvidence],
    top_k: int,
) -> list[SearchResult]:
    candidates = build_content_candidates(results, content_evidence)
    if not candidates:
        return results[:top_k]

    attempted_deployments: list[str] = []
    last_deployment_error: Exception | None = None
    for deployment in deployment_candidates("deep"):
        attempted_deployments.append(deployment)
        config = get_azure_openai_config("deep", deployment=deployment)
        try:
            return request_content_rerank(query, candidates, top_k, config, results)
        except Exception as error:
            if is_deployment_not_found_error(error):
                last_deployment_error = error
                continue
            raise

    if last_deployment_error:
        raise RuntimeError(
            format_deployment_not_found_message(
                "deep",
                attempted_deployments,
                last_deployment_error,
            )
        )

    raise RuntimeError("Azure OpenAI deep deployment is not configured.")


def request_content_rerank(
    query: str,
    candidates: list[dict[str, Any]],
    top_k: int,
    config: Any,
    current_results: list[SearchResult],
) -> list[SearchResult]:

    try:
        from openai import AzureOpenAI
    except ImportError as error:
        raise RuntimeError(
            "Deep content reranking needs the OpenAI SDK. Run: pip3 install -r requirements.txt"
        ) from error

    client = AzureOpenAI(
        api_version=config.api_version,
        azure_endpoint=config.endpoint,
        api_key=config.api_key,
    )
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a document search investigator. Rank candidate files "
                    "using only the provided metadata, extracted text samples, "
                    "matched terms, and snippets. Do not invent unseen content. "
                    "Return strict JSON only."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "query": query,
                        "top_k": top_k,
                        "candidate_evidence": candidates,
                        "response_schema": {
                            "results": [
                                {
                                    "file_id": "string",
                                    "score": "number from 0 to 100",
                                    "reasons": [
                                        "short user-facing reason grounded in provided evidence"
                                    ],
                                }
                            ]
                        },
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        max_completion_tokens=2400,
        model=config.deployment,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    if not content:
        raise ValueError("Azure OpenAI returned an empty message")

    parsed = json.loads(content)
    rankings = parsed.get("results", [])
    if not isinstance(rankings, list):
        raise ValueError("Deep rerank response JSON must contain a results list")

    by_file_id = {str(result.record.get("file_id")): result for result in current_results}
    reranked: list[SearchResult] = []
    for ranking in rankings:
        if not isinstance(ranking, dict):
            continue
        file_id = str(ranking.get("file_id"))
        current = by_file_id.get(file_id)
        if not current:
            continue

        reasons = ranking.get("reasons", [])
        if not isinstance(reasons, list):
            reasons = [str(reasons)]
        positive_reasons = sanitize_reasons([str(reason) for reason in reasons])
        if not positive_reasons:
            positive_reasons = current.reasons

        reranked.append(
            SearchResult(
                record=current.record,
                score=float(ranking.get("score", current.score)),
                reasons=positive_reasons,
            )
        )

    return reranked[:top_k] or current_results[:top_k]


def build_content_candidates(
    results: list[SearchResult],
    content_evidence: list[ContentEvidence],
) -> list[dict[str, Any]]:
    evidence_by_file_id = {
        evidence.file_id: evidence
        for evidence in content_evidence
        if not evidence.error and (evidence.text_samples or evidence.snippets)
    }

    candidates: list[dict[str, Any]] = []
    for result in results:
        record = result.record
        file_id = str(record.get("file_id"))
        evidence = evidence_by_file_id.get(file_id)
        if not evidence:
            continue

        candidates.append(
            {
                "file_id": file_id,
                "filename": record.get("filename"),
                "title": record.get("title"),
                "extension": record.get("extension"),
                "relative_path": record.get("relative_path"),
                "folder_path": record.get("folder_path"),
                "local_score": result.score,
                "current_reasons": result.reasons[:8],
                "matched_terms": evidence.matched_terms,
                "snippets": evidence.snippets,
                "text_samples": evidence.text_samples,
                "inspected_pages": evidence.inspected_pages,
                "page_count": evidence.page_count,
                "ocr_used": evidence.ocr_used,
            }
        )

    return candidates
