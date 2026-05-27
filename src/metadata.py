"""Canonical file metadata plus derived search fields.

The generated index stores only stable facts. Search code can derive display
and ranking fields from the URI, so the index stays portable across local,
Azure Blob, and later OneDrive-style sources.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


DEFAULT_LOCAL_DATA_DIR = Path("data/raw")


def hydrate_records(
    records: list[dict[str, Any]],
    index_path: Path | None = None,
) -> list[dict[str, Any]]:
    local_data_dir = resolve_local_data_dir(index_path)
    return [hydrate_record(record, local_data_dir) for record in records]


def hydrate_record(record: dict[str, Any], local_data_dir: Path) -> dict[str, Any]:
    hydrated = dict(record)
    uri = str(
        hydrated.get("uri")
        or hydrated.get("relative_path")
        or hydrated.get("path")
        or ""
    )
    source = str(hydrated.get("source") or "local")
    path = Path(uri)
    parents = path.parts[:-1]
    extension = path.suffix.lower()
    size_bytes = hydrated.get("size_bytes", hydrated.get("file_size_bytes", 0))

    hydrated["source"] = source
    hydrated["uri"] = uri
    hydrated["relative_path"] = uri
    hydrated["filename"] = path.name
    hydrated["title"] = path.stem
    hydrated["extension"] = extension
    hydrated["folder_path"] = Path(*parents).as_posix() if parents else ""
    hydrated["parent_folder"] = parents[-1] if len(parents) >= 1 else None
    hydrated["grandparent_folder"] = parents[-2] if len(parents) >= 2 else None
    hydrated["file_size_bytes"] = int(size_bytes or 0)
    hydrated["size_bytes"] = int(size_bytes or 0)
    hydrated["file_size_label"] = format_file_size(int(size_bytes or 0))
    hydrated["type_label"] = infer_type_label(extension)

    if source == "local" and not hydrated.get("absolute_path") and uri:
        hydrated["absolute_path"] = str((local_data_dir / uri).resolve())

    return hydrated


def resolve_local_data_dir(index_path: Path | None = None) -> Path:
    configured = os.getenv("ADS_LOCAL_DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()

    if index_path:
        project_root = index_path.resolve().parent.parent
        return (project_root / DEFAULT_LOCAL_DATA_DIR).resolve()

    return DEFAULT_LOCAL_DATA_DIR.resolve()


def infer_type_label(extension: str) -> str:
    if extension in {".ppt", ".pptx"}:
        return "slide"
    if extension == ".pdf":
        return "pdf"
    if extension in {".doc", ".docx"}:
        return "document"
    if extension in {".xlsx", ".xlsm", ".xls"}:
        return "spreadsheet"
    return "unknown"


def format_file_size(size_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    size = float(size_bytes)

    for unit in units:
        if size < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024

    return f"{size_bytes} B"
