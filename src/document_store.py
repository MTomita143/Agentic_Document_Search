"""Resolve document URIs to local files for lazy inspection."""

from __future__ import annotations

import hashlib
import os
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_BLOB_CACHE_DIR = Path("indexes/blob_cache")


def resolve_document_path(record: dict[str, Any]) -> Path:
    source = str(record.get("source") or "local")
    if source == "local":
        return Path(str(record.get("absolute_path")))
    if source == "azure_blob":
        return download_blob_to_cache(record)
    raise RuntimeError(f"Unsupported document source: {source}")


def download_blob_to_cache(record: dict[str, Any]) -> Path:
    container_url = os.getenv("AZURE_BLOB_CONTAINER_URL", "").strip()
    if not container_url:
        raise RuntimeError(
            "Azure Blob document source needs AZURE_BLOB_CONTAINER_URL."
        )

    uri = str(record.get("uri") or record.get("relative_path") or "")
    if not uri:
        raise RuntimeError("Azure Blob record is missing uri.")

    cache_path = build_blob_cache_path(record)
    if cache_path.exists() and cache_path.stat().st_size == int(record.get("size_bytes", 0)):
        return cache_path

    blob_url = build_blob_url(container_url, uri)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(blob_url, timeout=60) as response:
            cache_path.write_bytes(response.read())
    except Exception as error:
        raise RuntimeError(f"Azure Blob download failed for {uri}: {error}") from error

    return cache_path


def build_blob_url(container_url: str, blob_name: str) -> str:
    parsed = urllib.parse.urlsplit(container_url.rstrip("/"))
    path = parsed.path.rstrip("/") + "/" + urllib.parse.quote(blob_name, safe="/")
    return urllib.parse.urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            path,
            parsed.query,
            parsed.fragment,
        )
    )


def build_blob_cache_path(record: dict[str, Any]) -> Path:
    configured = os.getenv("ADS_BLOB_CACHE_DIR", "").strip()
    cache_dir = Path(configured).expanduser() if configured else DEFAULT_BLOB_CACHE_DIR
    raw_key = "|".join(
        [
            str(record.get("source")),
            str(record.get("uri") or record.get("relative_path")),
            str(record.get("size_bytes") or record.get("file_size_bytes")),
            str(record.get("modified_time")),
        ]
    )
    digest = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:16]
    extension = Path(str(record.get("uri") or "")).suffix.lower()
    return cache_dir / f"{digest}{extension}"
