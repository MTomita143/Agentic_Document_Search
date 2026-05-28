"""Client for the optional GPU VM CLIP page-ranking worker."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from clip_prefilter import ClipEvidence
from search import SearchResult


DEFAULT_CLIP_WORKER_MODEL = "clip-ViT-B-32"
DEFAULT_CLIP_WORKER_TIMEOUT_SECONDS = 120


def missing_visual_worker_config() -> list[str]:
    missing: list[str] = []
    if not os.getenv("CLIP_WORKER_URL", "").strip():
        missing.append("CLIP_WORKER_URL")
    return missing


def rank_visual_pages_with_worker(
    query: str,
    results: list[SearchResult],
    max_files: int,
    max_pages_per_file: int,
    top_pages: int,
    model_name: str | None = None,
) -> tuple[list[ClipEvidence], dict[str, list[int]]]:
    worker_url = os.getenv("CLIP_WORKER_URL", "").strip().rstrip("/")
    if not worker_url:
        raise RuntimeError("CLIP_WORKER_URL is not configured.")

    payload = {
        "query": query,
        "candidates": [
            build_candidate_payload(result)
            for result in results[:max_files]
        ],
        "max_files": max_files,
        "max_pages_per_file": max_pages_per_file,
        "top_pages": top_pages,
        "model_name": model_name
        or os.getenv("CLIP_WORKER_MODEL", DEFAULT_CLIP_WORKER_MODEL),
    }
    response = post_json(
        url=f"{worker_url}/rank-pages",
        payload=payload,
        timeout_seconds=get_worker_timeout_seconds(),
    )

    selected_pages_by_file = {
        str(file_id): [int(page) for page in pages]
        for file_id, pages in response.get("selected_pages_by_file", {}).items()
    }
    evidence = [
        build_clip_evidence(item)
        for item in response.get("evidence", [])
    ]
    return evidence, selected_pages_by_file


def build_candidate_payload(result: SearchResult) -> dict[str, Any]:
    record = result.record
    uri = str(record.get("uri") or record.get("relative_path") or "")
    return {
        "file_id": str(record.get("file_id")),
        "uri": uri,
        "filename": record.get("filename") or uri.rsplit("/", 1)[-1],
        "extension": record.get("extension"),
        "size_bytes": record.get("size_bytes") or record.get("file_size_bytes"),
        "modified_time": record.get("modified_time"),
    }


def post_json(
    url: str,
    payload: dict[str, Any],
    timeout_seconds: int,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    api_key = os.getenv("CLIP_WORKER_API_KEY", "").strip()
    if api_key:
        headers["X-Clip-Worker-Key"] = api_key

    request = urllib.request.Request(
        url,
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"GPU CLIP worker returned HTTP {error.code}: {detail}"
        ) from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"GPU CLIP worker could not be reached: {error}") from error


def build_clip_evidence(item: dict[str, Any]) -> ClipEvidence:
    selected_pages = [int(page) for page in item.get("selected_pages", [])]
    error = item.get("error")
    observations = []
    if selected_pages:
        observations.append(
            "GPU CLIP selected likely visual pages: "
            + ", ".join(str(page) for page in selected_pages)
        )

    return ClipEvidence(
        file_id=str(item.get("file_id")),
        score=float(item.get("score") or 0),
        selected_pages=selected_pages,
        observations=observations,
        rendered_pages=int(item.get("rendered_pages") or 0),
        page_count=int(item.get("page_count") or 0),
        cache_hits=0,
        error=str(error) if error else None,
    )


def get_worker_timeout_seconds() -> int:
    raw_value = os.getenv("CLIP_WORKER_TIMEOUT_SECONDS", "").strip()
    if not raw_value:
        return DEFAULT_CLIP_WORKER_TIMEOUT_SECONDS

    try:
        return max(5, int(raw_value))
    except ValueError:
        return DEFAULT_CLIP_WORKER_TIMEOUT_SECONDS
