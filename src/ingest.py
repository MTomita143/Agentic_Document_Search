"""
Build a lightweight file-level metadata index.

The saved index is intentionally small and portable. Richer fields such as
filename, extension, parent folder, display size, and local absolute path are
derived when the app loads the index.

```json
{
  "file_id": "file_xxxxx",
  "source": "local",
  "uri": "CompanyA/Reports/Q4_Revenue_Update.pptx",
  "size_bytes": 1234567,
  "modified_time": "2026-05-27T12:00:00+00:00"
}
```

Usage:
    python src/ingest.py --data-dir data/raw --output indexes/files_index.json
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from blob_url import build_blob_listing_url, parse_blob_container_url


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".pptx",
    ".ppt",
    ".docx",
    ".doc",
    ".xlsx",
    ".xlsm",
    ".xls",
}


def make_file_id(relative_path: str) -> str:
    """Create a stable short ID from the relative path."""
    digest = hashlib.md5(relative_path.encode("utf-8")).hexdigest()
    return f"file_{digest[:10]}"


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
        stat = path.stat()

        record = {
            "file_id": make_file_id(relative_path_str),
            "source": "local",
            "uri": relative_path_str,
            "size_bytes": stat.st_size,
            "modified_time": datetime.fromtimestamp(
                stat.st_mtime,
                timezone.utc,
            ).isoformat(),
        }

        records.append(record)

    return records



def redact_query(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, "SAS_REDACTED", parsed.fragment)
    )


def collect_azure_blob_metadata(container_url: str) -> list[dict[str, Any]]:
    """List supported blobs from an Azure Blob container URL with read/list access."""
    listing_url = build_blob_listing_url(container_url)
    try:
        with urllib.request.urlopen(listing_url, timeout=60) as response:
            body = response.read()
    except urllib.error.HTTPError as error:
        if error.code == 404:
            reference = parse_blob_container_url(container_url)
            raise RuntimeError(
                "Azure Blob container was not found. "
                "AZURE_BLOB_CONTAINER_URL must point to an existing container, "
                "not only a virtual folder name. "
                f"Parsed container URL: {redact_query(reference.container_url)}. "
                f"Parsed prefix: {reference.prefix or '(none)'}. "
                "Use https://ACCOUNT.blob.core.windows.net/CONTAINER?SAS, "
                "or https://ACCOUNT.blob.core.windows.net/CONTAINER/PREFIX?SAS "
                "when PREFIX is a folder inside that container."
            ) from error
        raise

    root = ET.fromstring(body)
    records: list[dict[str, Any]] = []

    for blob in root.findall(".//Blob"):
        name = blob.findtext("Name") or ""
        extension = Path(name).suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            continue

        properties = blob.find("Properties")
        size_text = properties.findtext("Content-Length") if properties is not None else "0"
        modified_time = properties.findtext("Last-Modified") if properties is not None else None

        records.append(
            {
                "file_id": make_file_id(name),
                "source": "azure_blob",
                "uri": name,
                "size_bytes": int(size_text or 0),
                "modified_time": modified_time,
            }
        )

    return records


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
        "--source",
        choices=["local", "azure-blob"],
        default="local",
        help="Where to scan documents from.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing source documents.",
    )
    parser.add_argument(
        "--blob-container-url",
        default=os.getenv("AZURE_BLOB_CONTAINER_URL", ""),
        help="Azure Blob container URL with SAS when --source azure-blob is used.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("indexes/files_index.json"),
        help="Output JSON path.",
    )

    args = parser.parse_args()

    if args.source == "local":
        if not args.data_dir.exists():
            raise FileNotFoundError(f"Data directory does not exist: {args.data_dir}")
        records = collect_file_metadata(args.data_dir)
    else:
        if not args.blob_container_url:
            raise ValueError("--source azure-blob needs --blob-container-url")
        records = collect_azure_blob_metadata(args.blob_container_url)
    save_json(records, args.output)

    print(f"Indexed {len(records)} files")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
