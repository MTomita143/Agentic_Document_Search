"""
Optional Azure AI Translator query expansion.

This module translates user queries into English when useful, so Japanese prompts
can still match English filenames, folders, and extracted document text.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass

from search import has_cjk


DEFAULT_TRANSLATOR_API_VERSION = "3.0"


@dataclass
class QueryTranslation:
    original_query: str
    expanded_query: str
    translated_query: str | None
    detected_language: str | None
    used: bool
    error: str | None = None


def should_translate_query(query: str) -> bool:
    return has_cjk(query)


def expand_query_with_translator(query: str) -> QueryTranslation:
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
        translated_query, detected_language = translate_text(query, to_language="en")
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
    if not os.getenv("AZURE_TRANSLATOR_ENDPOINT"):
        missing.append("AZURE_TRANSLATOR_ENDPOINT")
    if not os.getenv("AZURE_TRANSLATOR_KEY"):
        missing.append("AZURE_TRANSLATOR_KEY")
    return missing


def translate_text(text: str, to_language: str) -> tuple[str, str | None]:
    endpoint = str(os.getenv("AZURE_TRANSLATOR_ENDPOINT", "")).rstrip("/")
    key = str(os.getenv("AZURE_TRANSLATOR_KEY", ""))
    region = os.getenv("AZURE_TRANSLATOR_REGION")
    api_version = os.getenv(
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
        raise RuntimeError(f"Azure Translator request failed: {details}") from error
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
