"""
Optional Azure AI Translator query expansion.

This module expands user queries across Japanese and English when useful, so
Japanese prompts can match English files and English prompts can still find
Japanese documents.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass

from env_utils import get_env
from search import has_cjk


DEFAULT_TRANSLATOR_API_VERSION = "3.0"
DEFAULT_TRANSLATOR_ENDPOINT = "https://api.cognitive.microsofttranslator.com"


@dataclass
class QueryTranslation:
    original_query: str
    expanded_query: str
    translated_query: str | None
    detected_language: str | None
    used: bool
    error: str | None = None


def should_translate_query(query: str) -> bool:
    return bool(query.strip())


def target_language_for_query(query: str) -> str:
    return "en" if has_cjk(query) else "ja"


def expand_query_with_translator(query: str) -> QueryTranslation:
    target_language = target_language_for_query(query)
    if not should_translate_query(query):
        return QueryTranslation(
            original_query=query,
            expanded_query=query,
            translated_query=None,
            detected_language=None,
            used=False,
        )

    missing = missing_translator_config()
    if missing:
        return QueryTranslation(
            original_query=query,
            expanded_query=query,
            translated_query=None,
            detected_language=None,
            used=False,
            error="Azure Translator is not configured: " + ", ".join(missing),
        )

    try:
        translated_query, detected_language = translate_text(
            query,
            to_language=target_language,
        )
    except RuntimeError as error:
        return QueryTranslation(
            original_query=query,
            expanded_query=query,
            translated_query=None,
            detected_language=None,
            used=False,
            error=str(error),
        )

    if not translated_query or translated_query.strip() == query.strip():
        return QueryTranslation(
            original_query=query,
            expanded_query=query,
            translated_query=translated_query,
            detected_language=detected_language,
            used=False,
        )

    return QueryTranslation(
        original_query=query,
        expanded_query=f"{query}\n{translated_query}",
        translated_query=translated_query,
        detected_language=detected_language,
        used=True,
    )


def missing_translator_config() -> list[str]:
    missing = []
    if not get_translator_key():
        missing.append("AZURE_TRANSLATOR_KEY or AZURE_OPENAI_API_KEY")
    return missing


def translate_text(text: str, to_language: str) -> tuple[str, str | None]:
    endpoint = str(
        get_env("AZURE_TRANSLATOR_ENDPOINT", DEFAULT_TRANSLATOR_ENDPOINT)
    ).rstrip("/")
    key = str(get_translator_key())
    region = get_env("AZURE_TRANSLATOR_REGION")
    api_version = get_env(
        "AZURE_TRANSLATOR_API_VERSION",
        DEFAULT_TRANSLATOR_API_VERSION,
    )

    query = urllib.parse.urlencode(
        {
            "api-version": api_version,
            "to": to_language,
        }
    )
    url = f"{endpoint}/translate?{query}"
    body = json.dumps([{"text": text}], ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Ocp-Apim-Subscription-Key": key,
        "X-ClientTraceId": str(uuid.uuid4()),
    }
    if region:
        headers["Ocp-Apim-Subscription-Region"] = region

    request = urllib.request.Request(
        url,
        data=body,
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            "Azure Translator request failed. If you are using a multi-service "
            "Azure AI resource key, set AZURE_TRANSLATOR_REGION to the resource "
            f"region. Details: {details}"
        ) from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Azure Translator request failed: {error}") from error

    if not payload or not isinstance(payload, list):
        raise RuntimeError("Azure Translator returned an unexpected response")

    first = payload[0]
    translations = first.get("translations", [])
    if not translations:
        raise RuntimeError("Azure Translator returned no translations")

    detected_language = None
    detected = first.get("detectedLanguage")
    if isinstance(detected, dict):
        detected_language = detected.get("language")

    translated_text = str(translations[0].get("text", ""))
    return translated_text, detected_language


def get_translator_key() -> str | None:
    return get_env("AZURE_TRANSLATOR_KEY") or get_env("AZURE_OPENAI_API_KEY")
