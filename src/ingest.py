"""
Build a lightweight file-level metadata index.

```json
{
  "file_id": "file_xxxxx",
  "filename": "Q4_Revenue_Update.pptx",
  "title": "Q4_Revenue_Update",
  "extension": ".pptx",
  "relative_path": "CompanyA/Reports/Q4_Revenue_Update.pptx",
  "parent_folder": "Reports",
  "grandparent_folder": "CompanyA",
  "folder_path": "CompanyA/Reports",
  "type_label": "slide"
}
```

Usage:
    python src/ingest.py --data-dir data/raw --output indexes/files_index.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".pptx",
    ".ppt",
    ".docx",
    ".doc",
}


def make_file_id(relative_path: str) -> str:
    """Create a stable short ID from the relative path."""
    digest = hashlib.md5(relative_path.encode("utf-8")).hexdigest()
    return f"file_{digest[:10]}"


def get_parent_parts(relative_path: Path) -> dict[str, str | None]:
    """
    Extract folder information.

    Example:
        company_a/reports/Q4_Revenue_Update.pptx

    parent_folder      -> reports
    grandparent_folder -> company_a
    folder_path        -> company_a/reports
    """
    parents = relative_path.parts[:-1]

    parent_folder = parents[-1] if len(parents) >= 1 else None
    grandparent_folder = parents[-2] if len(parents) >= 2 else None
    folder_path = str(Path(*parents)) if parents else ""

    return {
        "parent_folder": parent_folder,
        "grandparent_folder": grandparent_folder,
        "folder_path": folder_path,
    }


def collect_file_metadata(data_dir: Path) -> list[dict[str, Any]]:
    """Scan files under data_dir and collect lightweight metadata."""
    records: list[dict[str, Any]] = []

    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue

        extension = path.suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            continue

        relative_path = path.relative_to(data_dir)
        relative_path_str = relative_path.as_posix()
        folder_info = get_parent_parts(relative_path)

        record = {
            "file_id": make_file_id(relative_path_str),
            "filename": path.name,
            "title": path.stem,
            "extension": extension,
            "relative_path": relative_path_str,
            "absolute_path": str(path.resolve()),
            "parent_folder": folder_info["parent_folder"],
            "grandparent_folder": folder_info["grandparent_folder"],
            "folder_path": folder_info["folder_path"],
            "type_label": infer_type_label(extension),
        }

        records.append(record)

    return records


def infer_type_label(extension: str) -> str:
    """Infer a simple non-LLM type label from extension."""
    if extension in {".ppt", ".pptx"}:
        return "slide"
    if extension == ".pdf":
        return "pdf"
    if extension in {".doc", ".docx"}:
        return "document"
    return "unknown"


def save_json(records: list[dict[str, Any]], output_path: Path) -> None:
    """Save metadata records as pretty JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a lightweight file metadata index."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing source documents.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("indexes/files_index.json"),
        help="Output JSON path.",
    )

    args = parser.parse_args()

    if not args.data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {args.data_dir}")

    records = collect_file_metadata(args.data_dir)
    save_json(records, args.output)

    print(f"Indexed {len(records)} files")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
