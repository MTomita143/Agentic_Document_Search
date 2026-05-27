"""
Local CLIP visual prefilter.

This layer ranks rendered candidate PDF pages with a local CLIP model before
any expensive visual verifier is called. The intended flow is:

metadata candidates -> local CLIP page ranking -> optional Azure Vision verifier

The CLIP dependency is optional because it downloads/runs a local ML model.
Install it with:
    .venv/bin/python -m pip install -r requirements-clip.txt
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from document_store import resolve_document_path
from search import SearchResult
from visual import choose_pages, render_page_png


DEFAULT_CLIP_CACHE_PATH = Path("indexes/clip_visual_cache.json")
DEFAULT_CLIP_MODEL_NAME = "clip-ViT-B-32"
DEFAULT_CLIP_MODEL_CACHE_DIR = Path("indexes/model_cache/huggingface")


@dataclass
class ClipPageMatch:
    file_id: str
    filename: str
    page_number: int
    similarity: float
    cache_hit: bool


@dataclass
class ClipEvidence:
    file_id: str
    score: float
    selected_pages: list[int]
    observations: list[str]
    rendered_pages: int
    page_count: int
    cache_hits: int
    error: str | None = None


def rank_visual_pages_with_clip(
    query: str,
    results: list[SearchResult],
    max_files: int,
    max_pages_per_file: int,
    top_pages: int,
    cache_path: Path = DEFAULT_CLIP_CACHE_PATH,
    model_name: str = DEFAULT_CLIP_MODEL_NAME,
    render_zoom: float = 1.0,
) -> tuple[list[ClipEvidence], dict[str, list[int]]]:
    model, image_loader = load_clip_runtime(model_name)
    text_embedding = encode_text(model, query)
    cache = load_cache(cache_path)

    all_matches: list[ClipPageMatch] = []
    page_counts: dict[str, int] = {}
    rendered_page_counts: dict[str, int] = {}
    cache_hits_by_file: dict[str, int] = {}
    errors: list[ClipEvidence] = []

    for result in results[:max_files]:
        record = result.record
        file_id = str(record.get("file_id"))
        filename = str(record.get("filename"))
        extension = str(record.get("extension", "")).lower()

        if extension != ".pdf":
            errors.append(
                ClipEvidence(
                    file_id=file_id,
                    score=0,
                    selected_pages=[],
                    observations=[],
                    rendered_pages=0,
                    page_count=0,
                    cache_hits=0,
                    error=f"CLIP visual prefilter for {extension} is not supported yet",
                )
            )
            continue

        try:
            page_matches, page_count, rendered_pages, cache_hits = score_pdf_pages(
                query_embedding=text_embedding,
                record=record,
                model=model,
                image_loader=image_loader,
                cache=cache,
                max_pages=max_pages_per_file,
                model_name=model_name,
                render_zoom=render_zoom,
            )
        except Exception as error:
            errors.append(
                ClipEvidence(
                    file_id=file_id,
                    score=0,
                    selected_pages=[],
                    observations=[],
                    rendered_pages=0,
                    page_count=0,
                    cache_hits=0,
                    error=str(error),
                )
            )
            continue

        page_counts[file_id] = page_count
        rendered_page_counts[file_id] = rendered_pages
        cache_hits_by_file[file_id] = cache_hits
        all_matches.extend(page_matches)

    selected_matches = sorted(
        all_matches,
        key=lambda match: match.similarity,
        reverse=True,
    )[:top_pages]
    selected_by_file: dict[str, list[int]] = {}

    for match in selected_matches:
        selected_by_file.setdefault(match.file_id, []).append(match.page_number)

    evidence_by_file: dict[str, ClipEvidence] = {}
    for match in selected_matches:
        current = evidence_by_file.get(match.file_id)
        observation = (
            f"page {match.page_number} looked visually relevant "
            f"to the query by CLIP similarity {match.similarity:.3f}"
        )
        score = max(match.similarity, 0.0) * 20

        if current is None:
            evidence_by_file[match.file_id] = ClipEvidence(
                file_id=match.file_id,
                score=score,
                selected_pages=[match.page_number],
                observations=[observation],
                rendered_pages=rendered_page_counts.get(match.file_id, 0),
                page_count=page_counts.get(match.file_id, 0),
                cache_hits=cache_hits_by_file.get(match.file_id, 0),
            )
        else:
            current.score += score
            current.selected_pages.append(match.page_number)
            current.observations.append(observation)

    save_cache(cache_path, cache)
    return list(evidence_by_file.values()) + errors, selected_by_file


def score_pdf_pages(
    query_embedding: list[float],
    record: dict[str, Any],
    model: Any,
    image_loader: Any,
    cache: dict[str, Any],
    max_pages: int,
    model_name: str,
    render_zoom: float,
) -> tuple[list[ClipPageMatch], int, int, int]:
    try:
        import fitz
    except ImportError as error:
        raise RuntimeError(
            "CLIP prefilter needs PyMuPDF. Run: .venv/bin/python -m pip install -r requirements.txt"
        ) from error

    file_id = str(record.get("file_id"))
    filename = str(record.get("filename"))
    path = resolve_document_path(record)
    page_matches: list[ClipPageMatch] = []
    cache_hits = 0

    with fitz.open(path) as document:
        page_count = document.page_count
        page_numbers = choose_pages(page_count=page_count, max_pages=max_pages)

        for page_number in page_numbers:
            cache_key = build_clip_cache_key(
                record=record,
                page_number=page_number,
                model_name=model_name,
                render_zoom=render_zoom,
            )
            cached = cache.get(cache_key)
            if cached:
                cache_hits += 1
                image_embedding = cached["embedding"]
            else:
                page = document.load_page(page_number - 1)
                image_bytes = render_page_png(page, render_zoom=render_zoom)
                image = image_loader(BytesIO(image_bytes)).convert("RGB")
                image_embedding = encode_image(model, image)
                cache[cache_key] = {
                    "file_id": file_id,
                    "filename": filename,
                    "page_number": page_number,
                    "embedding": image_embedding,
                }

            page_matches.append(
                ClipPageMatch(
                    file_id=file_id,
                    filename=filename,
                    page_number=page_number,
                    similarity=dot_product(query_embedding, image_embedding),
                    cache_hit=bool(cached),
                )
            )

    return page_matches, page_count, len(page_matches), cache_hits


def load_clip_runtime(model_name: str) -> tuple[Any, Any]:
    model_cache_dir = get_clip_model_cache_dir()
    try:
        model_cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise RuntimeError(
            "CLIP could not create its local model cache directory. "
            f"The current cache directory is {model_cache_dir}. "
            "Set ADS_CLIP_MODEL_CACHE_DIR to a writable project path."
        ) from error

    os.environ.setdefault("HF_HOME", str(model_cache_dir))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(model_cache_dir))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(model_cache_dir / "transformers"))

    try:
        from PIL import Image
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "CLIP visual prefilter needs optional dependencies. "
            "Run: .venv/bin/python -m pip install -r requirements-clip.txt"
        ) from error

    try:
        model = SentenceTransformer(
            model_name,
            cache_folder=str(model_cache_dir),
        )
    except OSError as error:
        raise RuntimeError(
            "CLIP could not load its local model cache. "
            f"The current cache directory is {model_cache_dir}. "
            "You can override it with ADS_CLIP_MODEL_CACHE_DIR."
        ) from error

    return model, Image.open


def get_clip_model_cache_dir() -> Path:
    configured = os.getenv("ADS_CLIP_MODEL_CACHE_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()
    return DEFAULT_CLIP_MODEL_CACHE_DIR


def encode_text(model: Any, text: str) -> list[float]:
    embedding = model.encode(text, normalize_embeddings=True)
    return to_float_list(embedding)


def encode_image(model: Any, image: Any) -> list[float]:
    embedding = model.encode(image, normalize_embeddings=True)
    return to_float_list(embedding)


def to_float_list(value: Any) -> list[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [float(item) for item in value]


def dot_product(left: list[float], right: list[float]) -> float:
    return sum(left_value * right_value for left_value, right_value in zip(left, right))


def build_clip_cache_key(
    record: dict[str, Any],
    page_number: int,
    model_name: str,
    render_zoom: float,
) -> str:
    raw_key = json.dumps(
        {
            "file_id": record.get("file_id"),
            "source": record.get("source"),
            "uri": record.get("uri") or record.get("relative_path"),
            "file_size_bytes": record.get("file_size_bytes"),
            "modified_time": record.get("modified_time"),
            "page_number": page_number,
            "model_name": model_name,
            "render_zoom": render_zoom,
        },
        sort_keys=True,
    )
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def load_cache(cache_path: Path) -> dict[str, Any]:
    if not cache_path.exists():
        return {}

    with cache_path.open("r", encoding="utf-8") as f:
        loaded = json.load(f)

    if isinstance(loaded, dict):
        return loaded
    return {}


def save_cache(cache_path: Path, cache: dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
