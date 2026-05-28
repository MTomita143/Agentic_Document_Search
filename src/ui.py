"""
Streamlit prototype for Agentic Document Search.

Run locally:
    .venv/bin/streamlit run src/ui.py
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import streamlit as st

from agent import run_agent
from document_store import resolve_document_path
from ingest import collect_azure_blob_metadata, collect_file_metadata, save_json
from search import DEFAULT_INDEX_PATH, load_environment


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FILE_TYPE_CHOICES = {
    "excel": {
        "label": "Excel",
        "icon": "▦",
        "extensions": {".xlsx", ".xlsm", ".xls"},
    },
    "word": {
        "label": "Word",
        "icon": "□",
        "extensions": {".docx", ".doc"},
    },
    "powerpoint": {
        "label": "PowerPoint",
        "icon": "▣",
        "extensions": {".pptx", ".ppt"},
    },
    "pdf": {
        "label": "PDF",
        "icon": "▤",
        "extensions": {".pdf"},
    },
}
VISUAL_FILE_TYPE_KEYS = {"pdf"}
TEXT_HEAVY_FILE_TYPE_KEYS = {"excel", "word"}


def main() -> None:
    load_environment()
    configure_page()

    st.title("Agentic Document Search")
    st.caption("Find documents the way you remember them.")

    selected_file_types = render_file_type_buttons()
    selected_extensions = extensions_for_file_types(selected_file_types)

    query = st.text_input(
        "What do you remember?",
        placeholder="Example: I remember a slide with a blue graph",
    )

    with st.sidebar:
        st.header("Search Settings")
        render_index_controls()

        search_mode = st.radio(
            "Search mode",
            options=["instant", "reasoning", "visual", "auto"],
            format_func=format_search_mode,
            index=0,
            help="Choose how much the agent should inspect before answering.",
        )
        settings = preset_settings(search_mode)
        st.caption(mode_summary(search_mode))

        with st.expander("⚙️ Customize mode"):
            if search_mode == "auto":
                st.caption(
                    "🧭 Auto mode lets Semantic Kernel choose these settings at run time."
                )
            use_llm = st.toggle(
                "🧠 Azure OpenAI rerank",
                value=settings["mode"] == "llm",
                key=f"{search_mode}_use_llm",
                help="Uses your Foundry/Azure OpenAI deployment and may cost money.",
            )
            use_text = st.toggle(
                "📄 Text inspection",
                value=settings["content_mode"] != "never",
                key=f"{search_mode}_use_text",
                help="Inspects extracted text from top candidate files.",
            )
            use_translator = st.toggle(
                "🌐 Azure Translator",
                value=settings["translator_mode"] == "auto",
                key=f"{search_mode}_use_translator",
                help="Expands Japanese queries into English for mixed-language search.",
            )
            use_azure_ai_search = st.toggle(
                "🔎 Azure AI Search",
                value=settings["azure_ai_search_mode"] == "auto",
                key=f"{search_mode}_use_azure_ai_search",
                help="Uses Azure AI Search as an optional retrieval tool.",
            )
            use_clip = st.toggle(
                "📎 CLIP prefilter",
                value=settings["visual_prefilter"] == "clip",
                key=f"{search_mode}_use_clip",
                help="Ranks rendered pages locally with CLIP before visual verification.",
            )
            use_vision = st.toggle(
                "👁️ Azure visual inspection",
                value=settings["visual_mode"] == "azure",
                key=f"{search_mode}_use_vision",
                help="Sends selected rendered pages to Azure AI Vision and may cost money.",
            )

            st.divider()
            st.caption("🛠️ Limits")
            top_k = st.slider(
                "Results",
                min_value=1,
                max_value=6,
                value=3,
                key=f"{search_mode}_top_k",
            )
            candidate_pool_size = st.slider(
                "LLM candidate pool",
                min_value=1,
                max_value=20,
                value=settings["candidate_pool_size"],
                key=f"{search_mode}_candidate_pool_size",
                help="Only used for Azure OpenAI reranking.",
            )
            if use_llm and candidate_pool_size < top_k:
                candidate_pool_size = top_k
                st.caption(
                    f"🧠 LLM candidate pool adjusted to {candidate_pool_size} "
                    "so it can return the requested number of results."
                )
            max_inspected_files = st.slider(
                "Text files to inspect",
                1,
                10,
                settings["max_inspected_files"],
                key=f"{search_mode}_max_inspected_files",
            )
            max_pages_per_file = st.slider(
                "Text pages per file",
                1,
                200,
                settings["max_pages_per_file"],
                key=f"{search_mode}_max_pages_per_file",
            )
            max_visual_files = st.slider(
                "Visual files to inspect",
                1,
                6,
                settings["max_visual_files"],
                key=f"{search_mode}_max_visual_files",
            )
            max_visual_pages_per_file = st.slider(
                "Visual pages per file",
                1,
                24,
                settings["max_visual_pages_per_file"],
                key=f"{search_mode}_max_visual_pages_per_file",
            )
            max_clip_pages = st.slider(
                "CLIP pages for Vision",
                1,
                60,
                settings["max_clip_pages"],
                key=f"{search_mode}_max_clip_pages",
            )
            if use_clip:
                requested_clip_pages = max_clip_pages
                minimum_clip_pages = (
                    max_visual_files * max_visual_pages_per_file
                    if use_vision
                    else max_visual_pages_per_file
                )
                max_clip_pages = max(max_clip_pages, minimum_clip_pages)
                st.caption(
                    f"📎 CLIP will select up to {max_clip_pages} pages; "
                    f"Azure Vision will inspect at most {max_visual_pages_per_file} per file."
                )
                if max_clip_pages != requested_clip_pages:
                    st.caption(
                        "📎 CLIP page count was auto-adjusted so the visual "
                        "inspection budget does not exceed the CLIP funnel."
                    )

        mode = "llm" if use_llm else "local"
        content_mode = "auto" if use_text else "never"
        translator_mode = "auto" if use_translator else "never"
        azure_ai_search_mode = "auto" if use_azure_ai_search else "never"
        visual_prefilter = "clip" if use_clip else "none"
        visual_mode = "azure" if use_vision else "never"
        orchestration_mode = (
            "semantic-kernel" if search_mode == "auto" else "manual"
        )

        if search_mode in {"auto", "visual"} and selected_file_types <= TEXT_HEAVY_FILE_TYPE_KEYS:
            mode = "llm"
            visual_mode = "never"
            visual_prefilter = "none"
            st.caption(
                "Word/Excel-only search uses reasoning over extracted text and skips visual inspection."
            )
        elif not selected_file_types & VISUAL_FILE_TYPE_KEYS:
            visual_mode = "never"
            visual_prefilter = "none"
            st.caption("Visual inspection is skipped because no visual-friendly file type is selected.")

    show_cost_notice(
        orchestration_mode,
        mode,
        translator_mode,
        azure_ai_search_mode,
        visual_mode,
        visual_prefilter,
        max_visual_files,
        max_visual_pages_per_file,
        max_clip_pages,
    )

    search_clicked = st.button("Search", type="primary")

    if not selected_file_types:
        st.warning("Choose at least one file type.")
        return

    if not query:
        if search_clicked:
            st.warning("Enter a search memory first.")
        show_empty_state()
        return

    index_path = PROJECT_ROOT / DEFAULT_INDEX_PATH
    content_cache_path = PROJECT_ROOT / "indexes/content_cache.json"
    visual_cache_path = PROJECT_ROOT / "indexes/visual_cache.json"
    clip_cache_path = PROJECT_ROOT / "indexes/clip_visual_cache.json"
    search_memory_path = PROJECT_ROOT / "indexes/search_memory.json"

    if not search_clicked:
        show_cached_response()
        return

    with st.spinner("Running agentic search..."):
        try:
            response = run_agent(
                query=query,
                index_path=index_path,
                mode=mode,
                top_k=top_k,
                candidate_pool_size=candidate_pool_size,
                content_mode=content_mode,
                translator_mode=translator_mode,
                max_inspected_files=max_inspected_files,
                max_pages_per_file=max_pages_per_file,
                max_chars_per_file=30000,
                content_cache_path=content_cache_path,
                visual_mode=visual_mode,
                visual_prefilter=visual_prefilter,
                max_visual_files=max_visual_files,
                max_visual_pages_per_file=max_visual_pages_per_file,
                max_clip_pages=max_clip_pages,
                visual_cache_path=visual_cache_path,
                clip_cache_path=clip_cache_path,
                orchestration_mode=orchestration_mode,
                azure_ai_search_mode=azure_ai_search_mode,
                search_memory_path=search_memory_path,
                allowed_extensions=selected_extensions,
            )
        except Exception as error:
            st.error(str(error))
            return

    st.session_state.last_response = response
    st.session_state.last_query = query
    show_agent_steps(response)
    show_results(response)


def show_cached_response() -> None:
    response = st.session_state.get("last_response")
    if not response:
        return

    st.caption(f"Showing previous result for: {st.session_state.get('last_query', '')}")
    show_agent_steps(response)
    show_results(response)


def render_file_type_buttons() -> set[str]:
    st.caption("File type")
    if "selected_file_types" not in st.session_state:
        st.session_state.selected_file_types = list(FILE_TYPE_CHOICES)

    selected = set(st.session_state.selected_file_types)
    columns = st.columns(4)
    for column, (file_type, config) in zip(columns, FILE_TYPE_CHOICES.items()):
        is_selected = file_type in selected
        button_label = f"{config['icon']} {config['label']}"
        with column:
            clicked = st.button(
                button_label,
                key=f"file_type_{file_type}",
                use_container_width=True,
                type="primary" if is_selected else "secondary",
            )
        if clicked:
            if is_selected and len(selected) > 1:
                selected.remove(file_type)
            elif not is_selected:
                selected.add(file_type)
            st.session_state.selected_file_types = [
                key
                for key in FILE_TYPE_CHOICES
                if key in selected
            ]
            rerun()

    return selected


def extensions_for_file_types(file_types: set[str]) -> set[str]:
    extensions: set[str] = set()
    for file_type in file_types:
        extensions.update(FILE_TYPE_CHOICES[file_type]["extensions"])
    return extensions


def rerun() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def render_index_controls() -> None:
    with st.expander("🗂️ Index"):
        st.caption(
            "Refresh after documents are added, deleted, renamed, or moved."
        )
        local_clicked = st.button(
            "Refresh local index",
            use_container_width=True,
        )
        if local_clicked:
            refresh_local_index()

        if os.getenv("AZURE_BLOB_CONTAINER_URL"):
            blob_clicked = st.button(
                "Refresh Azure Blob index",
                use_container_width=True,
            )
            if blob_clicked:
                refresh_blob_index()


def refresh_local_index() -> None:
    data_dir = PROJECT_ROOT / "data/raw"
    index_path = PROJECT_ROOT / DEFAULT_INDEX_PATH
    try:
        records = collect_file_metadata(data_dir)
        save_json(records, index_path)
    except Exception as error:
        st.error(f"Could not refresh local index: {error}")
        return

    st.success(f"Refreshed local index with {len(records)} files.")


def refresh_blob_index() -> None:
    container_url = os.getenv("AZURE_BLOB_CONTAINER_URL", "")
    index_path = PROJECT_ROOT / DEFAULT_INDEX_PATH
    try:
        records = collect_azure_blob_metadata(container_url)
        save_json(records, index_path)
    except Exception as error:
        st.error(f"Could not refresh Azure Blob index: {error}")
        return

    st.success(f"Refreshed Azure Blob index with {len(records)} files.")


def configure_page() -> None:
    st.set_page_config(
        page_title="Agentic Document Search",
        page_icon="🔎",
        layout="wide",
    )


def format_search_mode(value: str) -> str:
    return {
        "instant": "⚡ Instant",
        "reasoning": "🧠 Reasoning",
        "visual": "👁️ Visual",
        "auto": "🧭 Auto",
    }[value]


def mode_summary(value: str) -> str:
    return {
        "instant": "Metadata plus local text inspection. No Azure AI calls.",
        "reasoning": "Instant search plus Azure OpenAI reranking.",
        "visual": "Reasoning search plus CLIP prefilter and Azure visual inspection.",
        "auto": "Semantic Kernel chooses the search strategy.",
    }[value]


def preset_settings(value: str) -> dict[str, object]:
    presets = {
        "instant": {
            "mode": "local",
            "content_mode": "auto",
            "translator_mode": "never",
            "azure_ai_search_mode": "never",
            "visual_mode": "never",
            "visual_prefilter": "none",
            "candidate_pool_size": 6,
            "max_inspected_files": 6,
            "max_pages_per_file": 50,
            "max_visual_files": 3,
            "max_visual_pages_per_file": 5,
            "max_clip_pages": 10,
        },
        "reasoning": {
            "mode": "llm",
            "content_mode": "auto",
            "translator_mode": "auto",
            "azure_ai_search_mode": "never",
            "visual_mode": "never",
            "visual_prefilter": "none",
            "candidate_pool_size": 6,
            "max_inspected_files": 6,
            "max_pages_per_file": 50,
            "max_visual_files": 3,
            "max_visual_pages_per_file": 5,
            "max_clip_pages": 10,
        },
        "visual": {
            "mode": "llm",
            "content_mode": "auto",
            "translator_mode": "auto",
            "azure_ai_search_mode": "never",
            "visual_mode": "azure",
            "visual_prefilter": "clip",
            "candidate_pool_size": 6,
            "max_inspected_files": 6,
            "max_pages_per_file": 50,
            "max_visual_files": 3,
            "max_visual_pages_per_file": 5,
            "max_clip_pages": 10,
        },
        "auto": {
            "mode": "local",
            "content_mode": "auto",
            "translator_mode": "auto",
            "azure_ai_search_mode": "auto",
            "visual_mode": "never",
            "visual_prefilter": "none",
            "candidate_pool_size": 6,
            "max_inspected_files": 6,
            "max_pages_per_file": 50,
            "max_visual_files": 3,
            "max_visual_pages_per_file": 5,
            "max_clip_pages": 10,
        },
    }
    return presets[value]


def show_empty_state() -> None:
    st.info("Try a vague memory like: “Find the report about open source.”")
    st.markdown(
        """
        **Good demo queries**

        - `Find the investor presentation deck`
        - `Find a report about open source`
        - `Find the document that talks about telecom`
        - `I remember a slide with a blue graph`
        - `売上分析のレポートを探して`
        - `青いグラフがある資料`
        """
    )


def show_cost_notice(
    orchestration_mode: str,
    mode: str,
    translator_mode: str,
    azure_ai_search_mode: str,
    visual_mode: str,
    visual_prefilter: str,
    max_visual_files: int,
    max_visual_pages_per_file: int,
    max_clip_pages: int,
) -> None:
    if orchestration_mode == "semantic-kernel":
        st.warning("Semantic Kernel auto mode may use Azure OpenAI to choose a search strategy.")

    if mode == "llm":
        st.warning("Azure OpenAI reranking is enabled. This may use paid tokens.")

    if translator_mode == "auto":
        st.info("Azure Translator query expansion is enabled for Japanese prompts.")

    if azure_ai_search_mode == "auto":
        st.info("Azure AI Search may be used as an optional retrieval tool.")

    if visual_prefilter == "clip":
        st.info(
            "CLIP prefilter is local. First use may require installing optional dependencies "
            "and downloading the CLIP model."
        )

    if visual_mode == "azure":
        max_calls = estimate_vision_calls(
            visual_prefilter,
            max_visual_files,
            max_visual_pages_per_file,
            max_clip_pages,
        )
        st.warning(
            f"Azure Vision is enabled. This run can make up to {max_calls} image-analysis calls."
        )
    elif visual_mode == "auto":
        max_calls = estimate_vision_calls(
            visual_prefilter,
            max_visual_files,
            max_visual_pages_per_file,
            max_clip_pages,
        )
        st.info(
            "Azure Vision may run only when the query has visual clues. "
            f"If it runs, the current maximum is {max_calls} image-analysis calls."
        )


def estimate_vision_calls(
    visual_prefilter: str,
    max_visual_files: int,
    max_visual_pages_per_file: int,
    max_clip_pages: int,
) -> int:
    if visual_prefilter == "clip":
        return max_clip_pages
    return max_visual_files * max_visual_pages_per_file


def show_agent_steps(response: object) -> None:
    st.subheader("Agent Steps")
    for step in response.steps:
        with st.container(border=True):
            st.markdown(f"**{step.name}** · `{step.status}`")
            st.write(step.detail)


def show_results(response: object) -> None:
    st.subheader("Results")
    if not response.results:
        st.write("No matches found.")
        return

    for index, result in enumerate(response.results, start=1):
        record = result.record
        with st.container(border=True):
            col_main, col_score = st.columns([4, 1])
            with col_main:
                st.markdown(f"### {index}. {record.get('filename')}")
                st.write(record.get("relative_path"))
                st.caption(
                    f"{record.get('type_label')} · {record.get('file_size_label', 'unknown')}"
                )
            with col_score:
                st.metric("Score", f"{result.score:.1f}")

            st.markdown("**Why this matched**")
            for reason in result.reasons:
                st.write(f"- {reason}")

            show_page_preview(result)


def show_page_preview(result: object) -> None:
    record = result.record
    if str(record.get("extension", "")).lower() != ".pdf":
        return

    preview_pages = extract_preview_pages(result.reasons)
    preview_label = (
        "Show selected page"
        if preview_pages != [1]
        else "Show first page"
    )
    key = f"preview_{record.get('file_id')}"
    if st.button(preview_label, key=key):
        st.session_state[f"{key}_visible"] = not st.session_state.get(
            f"{key}_visible",
            False,
        )

    if not st.session_state.get(f"{key}_visible", False):
        return

    try:
        image_paths = render_preview_pages(record, preview_pages[:2])
    except Exception as error:
        st.warning(f"Could not render preview: {error}")
        return

    for image_path, page_number in image_paths:
        st.image(
            str(image_path),
            caption=f"Page {page_number}",
            use_container_width=True,
        )


def extract_preview_pages(reasons: list[str]) -> list[int]:
    pages: list[int] = []
    for reason in reasons:
        if "CLIP selected visual pages:" in reason:
            pages.extend(int(value) for value in re.findall(r"\d+", reason))
        else:
            pages.extend(
                int(value)
                for value in re.findall(r"\bpage\s+(\d+)\b", reason, flags=re.IGNORECASE)
            )

    unique_pages = []
    for page in pages:
        if page not in unique_pages and page > 0:
            unique_pages.append(page)

    return unique_pages or [1]


def render_preview_pages(
    record: dict[str, object],
    page_numbers: list[int],
) -> list[tuple[Path, int]]:
    try:
        import fitz
    except ImportError as error:
        raise RuntimeError("PDF preview needs PyMuPDF from requirements.txt") from error

    source_path = resolve_document_path(record)
    preview_dir = PROJECT_ROOT / "indexes/page_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    output: list[tuple[Path, int]] = []

    with fitz.open(source_path) as document:
        for page_number in page_numbers:
            if page_number < 1 or page_number > document.page_count:
                continue

            image_path = (
                preview_dir
                / f"{record.get('file_id')}-page-{page_number}.png"
            )
            if not image_path.exists():
                page = document.load_page(page_number - 1)
                pixmap = page.get_pixmap(matrix=fitz.Matrix(1.4, 1.4), alpha=False)
                pixmap.save(image_path)
            output.append((image_path, page_number))

    return output


if __name__ == "__main__":
    try:
        main()
    except ModuleNotFoundError as error:
        missing = error.name or "dependency"
        print(
            f"Missing dependency: {missing}. Run .venv/bin/python -m pip install -r requirements.txt",
            file=sys.stderr,
        )
        raise
