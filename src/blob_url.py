"""Azure Blob URL helpers.

Azure Blob Storage folders are virtual prefixes, not real containers. These
helpers let the app accept either a container SAS URL or a URL that includes a
prefix inside the container.
"""

from __future__ import annotations

from dataclasses import dataclass
import urllib.parse


@dataclass(frozen=True)
class BlobContainerReference:
    container_url: str
    prefix: str


def parse_blob_container_url(url: str) -> BlobContainerReference:
    parsed = urllib.parse.urlsplit(url.strip())
    path_parts = [part for part in parsed.path.split("/") if part]
    if not path_parts:
        return BlobContainerReference(container_url=url.strip(), prefix="")

    container_name = path_parts[0]
    prefix = "/".join(path_parts[1:]).strip("/")
    container_path = "/" + container_name
    container_url = urllib.parse.urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            container_path,
            parsed.query,
            parsed.fragment,
        )
    )
    return BlobContainerReference(container_url=container_url, prefix=prefix)


def build_blob_listing_url(container_url: str) -> str:
    reference = parse_blob_container_url(container_url)
    parsed = urllib.parse.urlsplit(reference.container_url)
    query_items = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    query_items.extend([("restype", "container"), ("comp", "list")])
    if reference.prefix:
        query_items.append(("prefix", reference.prefix.rstrip("/") + "/"))
    query = urllib.parse.urlencode(query_items)
    return urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, query, parsed.fragment)
    )


def build_blob_url(container_url: str, blob_name: str) -> str:
    reference = parse_blob_container_url(container_url)
    parsed = urllib.parse.urlsplit(reference.container_url.rstrip("/"))
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
