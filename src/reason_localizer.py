"""User-facing reason localization.

Search and visual inspection run best with bilingual/English-leaning internal
evidence, but the final "why this matched" text should follow the user's prompt
language. This module keeps file names and paths intact while rendering concise
Japanese reasons for Japanese prompts.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from azure_openai_config import (
    deployment_candidates,
    format_deployment_not_found_message,
    get_azure_openai_config,
    is_deployment_not_found_error,
    missing_azure_openai_config,
)
from search import SearchResult, dedupe, has_cjk


def should_localize_reasons_to_japanese(query: str, detected_language: str | None) -> bool:
    return has_cjk(query) or detected_language == "ja"


def localize_reasons_to_japanese(
    query: str,
    results: list[SearchResult],
    detected_language: str | None,
) -> None:
    if not should_localize_reasons_to_japanese(query, detected_language):
        return

    try:
        localized = localize_with_azure_openai(query, results)
    except Exception:
        localized = localize_with_templates(results)

    by_file_id = {
        str(item.get("file_id")): item.get("reasons", [])
        for item in localized
        if isinstance(item, dict)
    }

    for result in results:
        file_id = str(result.record.get("file_id"))
        reasons = by_file_id.get(file_id)
        if not isinstance(reasons, list):
            result.reasons = translate_reason_list_with_templates(result.reasons)
            continue
        cleaned = [str(reason).strip() for reason in reasons if str(reason).strip()]
        result.reasons = dedupe(cleaned)[:5] or translate_reason_list_with_templates(
            result.reasons
        )


def localize_with_azure_openai(
    query: str,
    results: list[SearchResult],
) -> list[dict[str, Any]]:
    localization_mode = os.getenv("ADS_REASON_LOCALIZATION", "auto").strip().lower()
    if localization_mode in {"template", "templates", "local", "off", "never"}:
        raise RuntimeError("Azure OpenAI reason localization is disabled.")

    if missing_azure_openai_config("fast"):
        raise RuntimeError("Azure OpenAI is not configured for reason localization.")

    try:
        from openai import AzureOpenAI
    except ImportError as error:
        raise RuntimeError("Reason localization needs the OpenAI SDK.") from error

    candidates = [
        {
            "file_id": str(result.record.get("file_id")),
            "filename": result.record.get("filename"),
            "relative_path": result.record.get("relative_path"),
            "reasons": result.reasons[:10],
        }
        for result in results
    ]
    attempted_deployments: list[str] = []
    last_deployment_error: Exception | None = None

    for deployment in deployment_candidates("fast"):
        attempted_deployments.append(deployment)
        config = get_azure_openai_config("fast", deployment=deployment)
        client = AzureOpenAI(
            api_version=config.api_version,
            azure_endpoint=config.endpoint,
            api_key=config.api_key,
        )
        try:
            response = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You rewrite document-search evidence into concise Japanese. "
                            "Use only the provided evidence. Do not invent content. "
                            "Do not translate file names, paths, product names, page numbers, "
                            "or company names. Avoid technical scores, cache details, and "
                            "model names unless the evidence specifically says a visual model "
                            "supported the match. Return strict JSON only."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "original_query": query,
                                "candidates": candidates,
                                "response_schema": {
                                    "results": [
                                        {
                                            "file_id": "string",
                                            "reasons": [
                                                "日本語で短い理由。ファイル名やパスはそのまま。"
                                            ],
                                        }
                                    ]
                                },
                            },
                            ensure_ascii=False,
                        ),
                    },
                ],
                max_completion_tokens=1800,
                model=config.deployment,
                response_format={"type": "json_object"},
            )
        except Exception as error:
            if is_deployment_not_found_error(error):
                last_deployment_error = error
                continue
            raise

        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("Azure OpenAI returned an empty localization response.")

        parsed = json.loads(content)
        localized = parsed.get("results", [])
        if isinstance(localized, list):
            return localized
        raise RuntimeError("Reason localization response must contain a results list.")

    if last_deployment_error:
        raise RuntimeError(
            format_deployment_not_found_message(
                "fast",
                attempted_deployments,
                last_deployment_error,
            )
        )
    raise RuntimeError("Azure OpenAI fast deployment is not configured.")


def localize_with_templates(results: list[SearchResult]) -> list[dict[str, Any]]:
    return [
        {
            "file_id": str(result.record.get("file_id")),
            "reasons": translate_reason_list_with_templates(result.reasons),
        }
        for result in results
    ]


def translate_reason_list_with_templates(reasons: list[str]) -> list[str]:
    translated = [translate_reason_with_template(reason) for reason in reasons]
    translated = [reason for reason in translated if reason]
    return dedupe(translated)[:5]


def translate_reason_with_template(reason: str) -> str:
    reason = " ".join(reason.strip().split())
    if not reason:
        return ""

    visual_pages = parse_visual_pages(reason)
    if visual_pages:
        return f"視覚的に関連しそうなページ: {visual_pages}"

    visual_skim = re.match(r"visual skim page (\d+) matched .*?: (.+)", reason)
    if visual_skim:
        page, detail = visual_skim.groups()
        return f"ページ{page}が視覚的な手がかりと一致しました（{translate_visual_detail(detail)}）。"

    if reason == "CLIP reinforced visual relevance":
        return "CLIPでも視覚的な関連性が補強されました。"

    if reason.lower().startswith("visual analysis matched"):
        terms = reason.split(":", 1)[1].strip() if ":" in reason else ""
        return f"Azure Vision の分析が手がかりと一致しました: {terms}" if terms else "Azure Vision の分析が手がかりと一致しました。"

    visual_observation = re.match(r"visual observation page (\d+): (.+)", reason)
    if visual_observation:
        page, detail = visual_observation.groups()
        return f"ページ{page}の視覚分析が手がかりと一致しました（{trim_japanese(detail)}）。"

    content_terms = re.match(r"content text matched: (.+)", reason)
    if content_terms:
        return f"本文に関連語が見つかりました: {content_terms.group(1)}"

    if reason.startswith("content snippet "):
        snippet = reason.replace("content snippet ", "", 1)
        return f"本文の抜粋が手がかりと一致しました: {trim_japanese(snippet)}"

    if reason.startswith("search memory:") or reason.startswith("previously appeared"):
        return "過去の似た検索でもこのファイルが候補に出ていました。"

    metadata_match = re.match(r"matched '([^']+)' in (.+)", reason)
    if metadata_match:
        term, field = metadata_match.groups()
        return f"{translate_metadata_field(field)}に「{term}」が含まれていました。"

    fuzzy_match = re.match(r"fuzzy matched '([^']+)' to '([^']+)' in (.+)", reason)
    if fuzzy_match:
        query_term, matched_term, field = fuzzy_match.groups()
        return f"{translate_metadata_field(field)}で「{query_term}」に近い「{matched_term}」が見つかりました。"

    if "metadata suggests presentation" in reason:
        return "メタデータからプレゼン資料らしいファイルだと判断しました。"

    if reason.startswith("text evidence matched terms:"):
        terms = reason.split(":", 1)[1].strip()
        return f"抽出した本文に関連語がありました: {terms}"

    if has_cjk(reason):
        return reason

    return trim_japanese(reason)


def parse_visual_pages(reason: str) -> str:
    if (
        reason.startswith("Visual page candidates:")
        or reason.startswith("CLIP selected visual pages:")
        or "視覚" in reason and "ページ" in reason
    ):
        pages = re.findall(r"\d+", reason)
        return ", ".join(pages)
    return ""


def translate_metadata_field(field: str) -> str:
    lower = field.lower()
    if "filename" in lower or "title" in lower:
        return "ファイル名またはタイトル"
    if "folder" in lower or "path" in lower:
        return "フォルダまたはパス"
    return "メタデータ"


def translate_visual_detail(detail: str) -> str:
    replacements = {
        "chart-like layout": "グラフらしいレイアウト",
        "table-like structure": "表らしい構造",
        "visual/layout-heavy page": "視覚要素の多いページ",
        "sampled visual page": "確認対象のページ",
    }
    for source, target in replacements.items():
        detail = detail.replace(source, target)
    detail = detail.replace("visible text matched", "表示テキストが一致:")
    return trim_japanese(detail)


def trim_japanese(text: str, max_chars: int = 130) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"
