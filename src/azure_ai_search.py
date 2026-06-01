"""Optional Azure AI Search retrieval adapter.

Azure AI Search is intentionally a tool, not the whole search engine. The
agent can ask it for candidates, then still decide whether to inspect metadata,
text, visual pages, or memory.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from env_utils import get_env
from search import SearchResult


DEFAULT_AZURE_SEARCH_API_VERSION = "2024-07-01"


@dataclass
class AzureAISearchConfig:
    endpoint: str
    key: str
    index_name: str
    api_version: str
    select_fields: str
    query_type: str
    semantic_configuration: str | None


def missing_azure_ai_search_config() -> list[str]:
    missing = []
    if not get_env("AZURE_AI_SEARCH_ENDPOINT"):
        missing.append("AZURE_AI_SEARCH_ENDPOINT")
    if not get_env("AZURE_AI_SEARCH_KEY"):
        missing.append("AZURE_AI_SEARCH_KEY")
    if not get_env("AZURE_AI_SEARCH_INDEX_NAME"):
        missing.append("AZURE_AI_SEARCH_INDEX_NAME")
    return missing


def load_azure_ai_search_config() -> AzureAISearchConfig:
    missing = missing_azure_ai_search_config()
    if missing:
        raise RuntimeError(
            "Azure AI Search is selected, but credentials are not configured: "
            + ", ".join(missing)
        )

    return AzureAISearchConfig(
        endpoint=str(get_env("AZURE_AI_SEARCH_ENDPOINT", "")).rstrip("/"),
        key=str(get_env("AZURE_AI_SEARCH_KEY", "")),
        index_name=str(get_env("AZURE_AI_SEARCH_INDEX_NAME", "")),
        api_version=get_env(
            "AZURE_AI_SEARCH_API_VERSION",
            DEFAULT_AZURE_SEARCH_API_VERSION,
        ),
        select_fields=get_env(
            "AZURE_AI_SEARCH_SELECT_FIELDS",
            "file_id,uri,relative_path,filename,title,content",
        ),
        query_type=str(get_env("AZURE_AI_SEARCH_QUERY_TYPE", "simple")),
        semantic_configuration=get_env("AZURE_AI_SEARCH_SEMANTIC_CONFIG"),
    )


def search_azure_ai_search(
    query: str,
    records: list[dict[str, Any]],
    top_k: int,
) -> list[SearchResult]:
    config = load_azure_ai_search_config()
    payload = build_search_payload(query, top_k, config)
    response = call_search_api(payload, config)
    return map_search_response_to_results(response, records)


def build_search_payload(
    query: str,
    top_k: int,
    config: AzureAISearchConfig,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "search": query,
        "top": top_k,
        "select": config.select_fields,
        "queryType": config.query_type,
    }
    if config.query_type == "semantic" and config.semantic_configuration:
        payload["semanticConfiguration"] = config.semantic_configuration
    return payload


def call_search_api(
    payload: dict[str, Any],
    config: AzureAISearchConfig,
) -> dict[str, Any]:
    index_name = urllib.parse.quote(config.index_name, safe="")
    url = (
        f"{config.endpoint}/indexes/{index_name}/docs/search"
        f"?api-version={urllib.parse.quote(config.api_version)}"
    )
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "api-key": config.key,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Azure AI Search request failed: {details}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Azure AI Search request failed: {error}") from error


def map_search_response_to_results(
    response: dict[str, Any],
    records: list[dict[str, Any]],
) -> list[SearchResult]:
    records_by_file_id = {
        str(record.get("file_id")): record
        for record in records
    }
    records_by_uri = {
        str(record.get("uri") or record.get("relative_path")): record
        for record in records
    }
    results: list[SearchResult] = []

    for item in response.get("value", []):
        record = find_matching_record(item, records_by_file_id, records_by_uri)
        if not record:
            continue

        score = float(item.get("@search.score", 0.0) or 0.0)
        reasons = [f"Azure AI Search returned this file with score {score:.2f}"]
        caption = extract_semantic_caption(item)
        if caption:
            reasons.append("Azure AI Search caption: " + caption)

        results.append(
            SearchResult(
                record=record,
                score=score,
                reasons=reasons,
            )
        )

    return results


def find_matching_record(
    item: dict[str, Any],
    records_by_file_id: dict[str, dict[str, Any]],
    records_by_uri: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    file_id = str(item.get("file_id") or item.get("id") or "")
    if file_id in records_by_file_id:
        return records_by_file_id[file_id]

    uri = str(item.get("uri") or item.get("relative_path") or item.get("path") or "")
    if uri in records_by_uri:
        return records_by_uri[uri]

    return None


def extract_semantic_caption(item: dict[str, Any]) -> str | None:
    captions = item.get("@search.captions")
    if not isinstance(captions, list) or not captions:
        return None

    text = captions[0].get("text")
    if not text:
        return None
    return str(text)
