"""Semantic Kernel auto-mode planner.

The Semantic Kernel path is optional at runtime. If the package or Azure OpenAI
planner config is missing, the app falls back to the same policy locally so the
search experience remains usable.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any

from search import has_cjk, normalize_text, tokenize


VISUAL_HINTS = {
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
    "図",
    "図表",
    "表",
}
CONTENT_HINTS = {
    "about",
    "analysis",
    "content",
    "explains",
    "mentions",
    "research",
    "says",
    "talks",
    "topic",
    "について",
    "内容",
    "分析",
    "調査",
    "説明",
}
METADATA_HINTS = {
    "deck",
    "directory",
    "doc",
    "document",
    "folder",
    "path",
    "pdf",
    "presentation",
    "report",
    "slide",
    "slides",
    "スライド",
    "フォルダ",
    "フォルダー",
    "プレゼン",
    "レポート",
    "報告書",
    "文書",
    "資料",
}


@dataclass
class SearchStrategyPlan:
    mode: str
    content_mode: str
    translator_mode: str
    visual_mode: str
    visual_prefilter: str
    azure_ai_search_mode: str
    used_semantic_kernel: bool
    rationale: str
    error: str | None = None


def plan_search_strategy(
    query: str,
    azure_ai_search_available: bool,
) -> SearchStrategyPlan:
    try:
        return asyncio.run(plan_with_semantic_kernel(query, azure_ai_search_available))
    except Exception as error:
        fallback = deterministic_plan(query, azure_ai_search_available)
        fallback.error = str(error)
        return fallback


async def plan_with_semantic_kernel(
    query: str,
    azure_ai_search_available: bool,
) -> SearchStrategyPlan:
    missing = missing_semantic_kernel_planner_config()
    if missing:
        raise RuntimeError(
            "Semantic Kernel planner is not configured: " + ", ".join(missing)
        )

    try:
        from semantic_kernel import Kernel
        from semantic_kernel.connectors.ai.open_ai import (
            AzureChatCompletion,
            AzureChatPromptExecutionSettings,
        )
        from semantic_kernel.connectors.ai.function_choice_behavior import (
            FunctionChoiceBehavior,
        )
        from semantic_kernel.functions import KernelArguments
    except ImportError as error:
        raise RuntimeError(
            "Semantic Kernel package is not installed. Install requirements-semantic-kernel.txt."
        ) from error

    kernel = Kernel()
    kernel.add_service(
        AzureChatCompletion(
            deployment_name=str(os.getenv("AZURE_OPENAI_DEPLOYMENT")),
            endpoint=str(os.getenv("AZURE_OPENAI_ENDPOINT")),
            api_key=str(os.getenv("AZURE_OPENAI_API_KEY")),
            api_version=str(os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")),
            service_id="planner",
        )
    )

    settings = AzureChatPromptExecutionSettings(
        service_id="planner",
        max_tokens=600,
        temperature=0,
        function_choice_behavior=FunctionChoiceBehavior.NoneInvoke(),
    )
    prompt = build_planner_prompt(query, azure_ai_search_available)
    result = await kernel.invoke_prompt(
        prompt,
        arguments=KernelArguments(settings=settings),
    )
    plan = parse_plan_json(str(result), query, azure_ai_search_available)
    plan.used_semantic_kernel = True
    return plan


def build_planner_prompt(query: str, azure_ai_search_available: bool) -> str:
    return (
        "You are the auto-mode planner for an agentic document search app.\n"
        "Choose the cheapest useful strategy. Return strict JSON only.\n"
        "Allowed values:\n"
        "- mode: local or llm\n"
        "- content_mode: auto, always, or never\n"
        "- translator_mode: auto or never\n"
        "- visual_mode: auto, azure, or never\n"
        "- visual_prefilter: clip or none\n"
        "- azure_ai_search_mode: auto or never\n"
        "Use visual_mode=azure and visual_prefilter=clip only for visual/layout/chart/table/image clues.\n"
        "Use translator_mode=auto for Japanese or mixed Japanese-English prompts.\n"
        "Use azure_ai_search_mode=auto only when it can help retrieve content-like candidates.\n"
        f"Azure AI Search configured: {azure_ai_search_available}\n"
        f"User query: {query}\n"
        "JSON schema: {\"mode\":\"local|llm\",\"content_mode\":\"auto|always|never\","
        "\"translator_mode\":\"auto|never\",\"visual_mode\":\"auto|azure|never\","
        "\"visual_prefilter\":\"clip|none\",\"azure_ai_search_mode\":\"auto|never\","
        "\"rationale\":\"short explanation\"}"
    )


def parse_plan_json(
    text: str,
    query: str,
    azure_ai_search_available: bool,
) -> SearchStrategyPlan:
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        parsed = json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        return deterministic_plan(query, azure_ai_search_available)

    fallback = deterministic_plan(query, azure_ai_search_available)
    return SearchStrategyPlan(
        mode=clean_choice(parsed.get("mode"), {"local", "llm"}, fallback.mode),
        content_mode=clean_choice(
            parsed.get("content_mode"),
            {"auto", "always", "never"},
            fallback.content_mode,
        ),
        translator_mode=clean_choice(
            parsed.get("translator_mode"),
            {"auto", "never"},
            fallback.translator_mode,
        ),
        visual_mode=clean_choice(
            parsed.get("visual_mode"),
            {"auto", "azure", "never"},
            fallback.visual_mode,
        ),
        visual_prefilter=clean_choice(
            parsed.get("visual_prefilter"),
            {"clip", "none"},
            fallback.visual_prefilter,
        ),
        azure_ai_search_mode=clean_choice(
            parsed.get("azure_ai_search_mode"),
            {"auto", "never"},
            fallback.azure_ai_search_mode,
        ),
        used_semantic_kernel=False,
        rationale=str(parsed.get("rationale") or fallback.rationale),
    )


def deterministic_plan(
    query: str,
    azure_ai_search_available: bool,
) -> SearchStrategyPlan:
    normalized = normalize_text(query)
    tokens = set(tokenize(query))
    has_visual = bool(tokens & VISUAL_HINTS) or any(term in normalized for term in VISUAL_HINTS)
    has_content = bool(tokens & CONTENT_HINTS) or any(term in normalized for term in CONTENT_HINTS)
    has_metadata = bool(tokens & METADATA_HINTS) or any(term in normalized for term in METADATA_HINTS)
    needs_translation = has_cjk(query)

    if has_visual:
        return SearchStrategyPlan(
            mode="llm",
            content_mode="auto",
            translator_mode="auto" if needs_translation else "never",
            visual_mode="azure",
            visual_prefilter="clip",
            azure_ai_search_mode="auto" if azure_ai_search_available else "never",
            used_semantic_kernel=False,
            rationale="Visual memory was detected, so use reasoning plus CLIP and Azure Vision.",
        )

    if has_content:
        return SearchStrategyPlan(
            mode="llm",
            content_mode="auto",
            translator_mode="auto" if needs_translation else "never",
            visual_mode="never",
            visual_prefilter="none",
            azure_ai_search_mode="auto" if azure_ai_search_available else "never",
            used_semantic_kernel=False,
            rationale="Content/topic memory was detected, so inspect candidate text and allow semantic retrieval.",
        )

    return SearchStrategyPlan(
        mode="local" if has_metadata else "llm",
        content_mode="auto",
        translator_mode="auto" if needs_translation else "never",
        visual_mode="never",
        visual_prefilter="none",
        azure_ai_search_mode="never",
        used_semantic_kernel=False,
        rationale="No visual clue was detected, so start with cheaper metadata and text signals.",
    )


def missing_semantic_kernel_planner_config() -> list[str]:
    missing = []
    for name in (
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_DEPLOYMENT",
    ):
        if not os.getenv(name):
            missing.append(name)
    return missing


def clean_choice(value: Any, allowed: set[str], default: str) -> str:
    if isinstance(value, str) and value in allowed:
        return value
    return default
