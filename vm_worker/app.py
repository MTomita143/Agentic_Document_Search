"""FastAPI CLIP page-ranking worker for an Azure GPU VM.

The App Service sends candidate document metadata. The worker downloads the
candidate PDFs from the shared Azure Blob container, renders a small page set,
ranks those pages against the query with CLIP, and returns top page numbers.
"""

from __future__ import annotations

import hashlib
import json
import os
from io import BytesIO
from pathlib import Path
from typing import Any
import urllib.parse
import urllib.request

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field


load_dotenv()

DEFAULT_MODEL_NAME = "clip-ViT-B-32"
DEFAULT_CACHE_DIR = Path("worker_cache")
DEFAULT_RENDER_ZOOM = 1.0

app = FastAPI(title="Agentic Document Search CLIP Worker")


class CandidateFile(BaseModel):
    file_id: str
    uri: str
    filename: str | None = None
    extension: str | None = None
    size_bytes: int | None = None
    modified_time: str | None = None


class RankPagesRequest(BaseModel):
    query: str
    candidates: list[CandidateFile]
    max_files: int = Field(default=3, ge=1, le=20)
    max_pages_per_file: int = Field(default=5, ge=1, le=120)
    top_pages: int = Field(default=10, ge=1, le=240)
    model_name: str = DEFAULT_MODEL_NAME
    render_zoom: float = Field(default=DEFAULT_RENDER_ZOOM, ge=0.5, le=3.0)


class PageMatch(BaseModel):
    file_id: str
    filename: str
    page_number: int
    similarity: float


class FileEvidence(BaseModel):
    file_id: str
    filename: str
    selected_pages: list[int]
    score: float
    rendered_pages: int
    page_count: int
    error: str | None = None


class RankPagesResponse(BaseModel):
    model_name: str
    device: str
    selected_pages_by_file: dict[str, list[int]]
    evidence: list[FileEvidence]
    matches: list[PageMatch]


def require_api_key(x_clip_worker_key: str | None = Header(default=None)) -> None:
    expected_key = os.getenv("CLIP_WORKER_API_KEY", "").strip()
    if expected_key and x_clip_worker_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid CLIP worker API key")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "device": get_device()}


@app.post("/rank-pages", dependencies=[Depends(require_api_key)])
def rank_pages(request: RankPagesRequest) -> RankPagesResponse:
    if not os.getenv("AZURE_BLOB_CONTAINER_URL", "").strip():
        raise HTTPException(
            status_code=500,
            detail="AZURE_BLOB_CONTAINER_URL is not configured on the worker",
        )

    model = load_clip_model(request.model_name)
    text_embedding = to_float_list(
        model.encode(request.query, normalize_embeddings=True)
    )

    all_matches: list[PageMatch] = []
    evidence_by_file: dict[str, FileEvidence] = {}

    for candidate in request.candidates[: request.max_files]:
        if (candidate.extension or Path(candidate.uri).suffix).lower() != ".pdf":
            evidence_by_file[candidate.file_id] = FileEvidence(
                file_id=candidate.file_id,
                filename=candidate.filename or Path(candidate.uri).name,
                selected_pages=[],
                score=0,
                rendered_pages=0,
                page_count=0,
                error="CLIP worker currently supports PDF page rendering only",
            )
            continue

        try:
            page_matches, page_count = score_candidate_pages(
                model=model,
                query_embedding=text_embedding,
                candidate=candidate,
                max_pages=request.max_pages_per_file,
                model_name=request.model_name,
                render_zoom=request.render_zoom,
            )
        except Exception as error:
            evidence_by_file[candidate.file_id] = FileEvidence(
                file_id=candidate.file_id,
                filename=candidate.filename or Path(candidate.uri).name,
                selected_pages=[],
                score=0,
                rendered_pages=0,
                page_count=0,
                error=str(error),
            )
            continue

        all_matches.extend(page_matches)
        evidence_by_file[candidate.file_id] = FileEvidence(
            file_id=candidate.file_id,
            filename=candidate.filename or Path(candidate.uri).name,
            selected_pages=[],
            score=0,
            rendered_pages=len(page_matches),
            page_count=page_count,
        )

    selected_matches = sorted(
        all_matches,
        key=lambda match: match.similarity,
        reverse=True,
    )[: request.top_pages]

    selected_pages_by_file: dict[str, list[int]] = {}
    for match in selected_matches:
        selected_pages_by_file.setdefault(match.file_id, []).append(match.page_number)
        evidence = evidence_by_file[match.file_id]
        evidence.selected_pages.append(match.page_number)
        evidence.score += max(match.similarity, 0.0) * 20

    return RankPagesResponse(
        model_name=request.model_name,
        device=get_device(),
        selected_pages_by_file=selected_pages_by_file,
        evidence=list(evidence_by_file.values()),
        matches=selected_matches,
    )


def score_candidate_pages(
    model: Any,
    query_embedding: list[float],
    candidate: CandidateFile,
    max_pages: int,
    model_name: str,
    render_zoom: float,
) -> tuple[list[PageMatch], int]:
    import fitz
    from PIL import Image

    local_path = download_blob_to_cache(candidate)
    page_matches: list[PageMatch] = []

    with fitz.open(local_path) as document:
        page_count = document.page_count
        for page_number in choose_pages(page_count, max_pages):
            image_embedding = get_or_create_page_embedding(
                model=model,
                document=document,
                candidate=candidate,
                page_number=page_number,
                model_name=model_name,
                render_zoom=render_zoom,
                image_loader=Image.open,
            )
            page_matches.append(
                PageMatch(
                    file_id=candidate.file_id,
                    filename=candidate.filename or Path(candidate.uri).name,
                    page_number=page_number,
                    similarity=dot_product(query_embedding, image_embedding),
                )
            )

    return page_matches, page_count


def get_or_create_page_embedding(
    model: Any,
    document: Any,
    candidate: CandidateFile,
    page_number: int,
    model_name: str,
    render_zoom: float,
    image_loader: Any,
) -> list[float]:
    import fitz

    cache_path = build_page_embedding_cache_path(candidate, page_number, model_name, render_zoom)
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))["embedding"]

    page = document.load_page(page_number - 1)
    image_bytes = page.get_pixmap(
        matrix=fitz.Matrix(render_zoom, render_zoom),
        alpha=False,
    ).tobytes("png")
    image = image_loader(BytesIO(image_bytes)).convert("RGB")
    embedding = to_float_list(model.encode(image, normalize_embeddings=True))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({"embedding": embedding}, ensure_ascii=False),
        encoding="utf-8",
    )
    return embedding


def load_clip_model(model_name: str) -> Any:
    from sentence_transformers import SentenceTransformer

    cache_dir = get_cache_dir() / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return SentenceTransformer(
        model_name,
        cache_folder=str(cache_dir),
        device=get_device(),
    )


def get_device() -> str:
    configured = os.getenv("CLIP_WORKER_DEVICE", "auto").strip().lower()
    if configured != "auto":
        return configured

    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def choose_pages(page_count: int, max_pages: int) -> list[int]:
    if page_count <= max_pages:
        return list(range(1, page_count + 1))

    if max_pages == 1:
        return [1]

    step = (page_count - 1) / (max_pages - 1)
    selected = [round(index * step) + 1 for index in range(max_pages)]
    unique_pages: list[int] = []
    for page_number in selected:
        page_number = min(max(page_number, 1), page_count)
        if page_number not in unique_pages:
            unique_pages.append(page_number)
    return unique_pages


def download_blob_to_cache(candidate: CandidateFile) -> Path:
    cache_path = build_blob_cache_path(candidate)
    if cache_path.exists() and (
        not candidate.size_bytes or cache_path.stat().st_size == candidate.size_bytes
    ):
        return cache_path

    blob_url = build_blob_url(candidate.uri)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(blob_url, timeout=120) as response:
        cache_path.write_bytes(response.read())
    return cache_path


def build_blob_url(blob_name: str) -> str:
    container_url = os.getenv("AZURE_BLOB_CONTAINER_URL", "").strip()
    parsed = urllib.parse.urlsplit(container_url.rstrip("/"))
    path = parsed.path.rstrip("/") + "/" + urllib.parse.quote(blob_name, safe="/")
    return urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, path, parsed.query, parsed.fragment)
    )


def build_blob_cache_path(candidate: CandidateFile) -> Path:
    raw_key = "|".join(
        [
            candidate.uri,
            str(candidate.size_bytes or ""),
            candidate.modified_time or "",
        ]
    )
    digest = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:16]
    extension = Path(candidate.uri).suffix.lower()
    return get_cache_dir() / "blobs" / f"{digest}{extension}"


def build_page_embedding_cache_path(
    candidate: CandidateFile,
    page_number: int,
    model_name: str,
    render_zoom: float,
) -> Path:
    raw_key = json.dumps(
        {
            "uri": candidate.uri,
            "size_bytes": candidate.size_bytes,
            "modified_time": candidate.modified_time,
            "page_number": page_number,
            "model_name": model_name,
            "render_zoom": render_zoom,
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
    return get_cache_dir() / "page_embeddings" / f"{digest}.json"


def get_cache_dir() -> Path:
    return Path(os.getenv("CLIP_WORKER_CACHE_DIR", DEFAULT_CACHE_DIR)).expanduser()


def to_float_list(value: Any) -> list[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [float(item) for item in value]


def dot_product(left: list[float], right: list[float]) -> float:
    return sum(left_value * right_value for left_value, right_value in zip(left, right))
