"""
Streamlit prototype for Agentic Document Search.

Run locally:
    .venv/bin/streamlit run src/ui.py
"""

from __future__ import annotations

import os
import re
import sys
from html import escape
from pathlib import Path

import streamlit as st

from agent import run_agent
from document_store import resolve_document_path
from ingest import collect_azure_blob_metadata, collect_file_metadata, save_json
from reason_localizer import translate_reason_with_template
from search import DEFAULT_INDEX_PATH, has_cjk, load_environment


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AZURE_RUNTIME_DIR_NAME = "agentic-document-search"
FILE_TYPE_CHOICES = {
    "excel": {
        "label": "Excel",
        "accent": "#16833a",
        "extensions": {".xlsx", ".xlsm", ".xls"},
    },
    "word": {
        "label": "Word",
        "accent": "#2563eb",
        "extensions": {".docx", ".doc"},
    },
    "powerpoint": {
        "label": "PowerPoint",
        "accent": "#eab30c",
        "extensions": {".pptx", ".ppt"},
    },
    "pdf": {
        "label": "PDF",
        "accent": "#fc3535",
        "extensions": {".pdf"},
    },
}
VISUAL_FILE_TYPE_KEYS = {"pdf"}
TEXT_HEAVY_FILE_TYPE_KEYS = {"excel", "word"}
FLOW_STEPS = [
    ("File Type Filter", "File types"),
    ("Translate Query", "Translate"),
    ("Understand Query", "Read request"),
    ("Semantic Kernel Auto Mode", "Plan route"),
    ("Metadata Search", "Names & folders"),
    ("Search Memory", "Recall"),
    ("Azure AI Search", "AI Search"),
    ("Inspect Content", "Read text"),
    ("Content LLM Rerank", "Deep reading"),
    ("CLIP Visual Prefilter", "Visual skim"),
    ("Inspect Visuals", "Check pages"),
    ("Return Answer", "Answer"),
]
FLOW_ALIASES = {
    "Metadata Search + LLM Rerank": "Metadata Search",
}
FLOW_LABELS = dict(FLOW_STEPS)


def main() -> None:
    load_environment()
    runtime_root = resolve_runtime_root()
    configure_runtime_environment(runtime_root)
    configure_page()
    inject_ui_styles()

    st.markdown(
        """
        <div class="ads-hero">
            <h1>Agentic Document Search</h1>
            <p class="ads-hero-main">Can't remember the filename?</p>
            <p class="ads-hero-sub">
                Describe what you remember — filename, content, charts, layout, or visual memory.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='ads-query-label'>Any clues about the document?</div>",
        unsafe_allow_html=True,
    )
    query = st.text_area(
        label="Any clues about the document?",
        placeholder="例: Apache関連のスライドで、昨年度の投資のフローチャートが載ってる",
        key="query_input",
        label_visibility="collapsed",
        height=112,
    )
    
    selected_file_types = render_file_type_buttons()
    selected_extensions = extensions_for_file_types(selected_file_types)

    with st.sidebar:
        st.header("Search Settings")
        render_index_controls(runtime_root)

        search_mode = st.radio(
            "Search mode",
            options=["auto", "instant", "reasoning", "visual"],
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
                "📖 Azure Translator",
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
                300,
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
                "Visual pages to skim per file",
                1,
                60,
                settings["max_clip_pages"],
                key=f"{search_mode}_max_clip_pages",
                help="How many pages per candidate file the hybrid visual funnel should skim before Azure Vision.",
            )
            if use_clip:
                requested_clip_pages = max_clip_pages
                minimum_clip_pages = max_visual_pages_per_file
                max_clip_pages = max(max_clip_pages, minimum_clip_pages)
                st.caption(
                    f"📎 The visual funnel will skim up to {max_clip_pages} pages per file; "
                    f"Azure Vision will inspect at most {max_visual_pages_per_file} per file."
                )
                if max_clip_pages != requested_clip_pages:
                    st.caption(
                        "📎 Skim budget was auto-adjusted so Azure Vision has enough selected pages."
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

        if (
            selected_file_types
            and search_mode in {"auto", "visual"}
            and selected_file_types <= TEXT_HEAVY_FILE_TYPE_KEYS
        ):
            mode = "llm"
            visual_mode = "never"
            visual_prefilter = "none"
            st.caption(
                "Word/Excel-only search uses reasoning over extracted text and skips visual inspection."
            )
        elif selected_file_types and not selected_file_types & VISUAL_FILE_TYPE_KEYS:
            visual_mode = "never"
            visual_prefilter = "none"
            st.caption("Visual inspection is skipped because no visual-friendly file type is selected.")

    _, search_col, _ = st.columns([1.2, 1, 1.2])
    with search_col:
        search_clicked = st.button(
            "Search",
            type="primary",
            key="search_button",
            use_container_width=True,
        )

    if not query:
        if search_clicked:
            show_prompt_warning()
        show_empty_state()
        return

    paths = resolve_app_paths(runtime_root)
    index_path = paths["index"]
    content_cache_path = paths["content_cache"]
    visual_cache_path = paths["visual_cache"]
    clip_cache_path = paths["clip_cache"]
    search_memory_path = paths["search_memory"]

    if not search_clicked:
        show_cached_response()
        return

    progress_placeholder = st.empty()
    with st.spinner("Preparing index and running agentic search..."):
        try:
            ensure_index_exists(index_path)
            render_progress_flow(
                progress_placeholder,
                steps=[],
                current_step="Semantic Kernel Auto Mode"
                if orchestration_mode == "semantic-kernel"
                else "Metadata Search",
            )

            def update_progress(steps: list[object], current_step: str | None) -> None:
                render_progress_flow(
                    progress_placeholder,
                    steps=steps,
                    current_step=current_step,
                )

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
                max_chars_per_file=200000,
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
                step_callback=update_progress,
            )
        except Exception as error:
            st.error(str(error))
            return

    progress_placeholder.empty()
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


def show_prompt_warning() -> None:
    st.markdown(
        """
        <div class="ads-inline-warning">
            Please type a search memory first.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_file_type_buttons() -> set[str]:
    st.markdown(
        """
        <div class='ads-file-type-spacer'></div>
        <div class='ads-file-caption'>
            Optional: narrow the search if you remember the file format
        </div>
        """,
        unsafe_allow_html=True,
    )
    if "selected_file_types" not in st.session_state:
        st.session_state.selected_file_types = []

    selected = set(st.session_state.selected_file_types)
    with st.container(key="file_type_filter"):
        columns = st.columns([1, 1, 1.4, 1])
        for column, (file_type, config) in zip(columns, FILE_TYPE_CHOICES.items()):
            is_selected = file_type in selected
            button_label = str(config["label"])
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


def extensions_for_file_types(file_types: set[str]) -> set[str] | None:
    if not file_types:
        return None

    extensions: set[str] = set()
    for file_type in file_types:
        extensions.update(FILE_TYPE_CHOICES[file_type]["extensions"])
    return extensions


def rerun() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def resolve_runtime_root() -> Path:
    configured = os.getenv("ADS_RUNTIME_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()

    if is_azure_app_service():
        home = Path(os.getenv("HOME", "/home")).expanduser()
        return home / AZURE_RUNTIME_DIR_NAME

    return PROJECT_ROOT


def is_azure_app_service() -> bool:
    return bool(
        os.getenv("WEBSITE_SITE_NAME")
        or os.getenv("WEBSITE_INSTANCE_ID")
        or os.getenv("WEBSITE_HOSTNAME")
    )


def configure_runtime_environment(runtime_root: Path) -> None:
    runtime_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(
        "ADS_BLOB_CACHE_DIR",
        str(runtime_root / "indexes/blob_cache"),
    )
    os.environ.setdefault(
        "ADS_CLIP_MODEL_CACHE_DIR",
        str(runtime_root / "indexes/model_cache/huggingface"),
    )


def resolve_app_paths(runtime_root: Path) -> dict[str, Path]:
    indexes_dir = runtime_root / "indexes"
    return {
        "index": runtime_root / DEFAULT_INDEX_PATH,
        "content_cache": indexes_dir / "content_cache.json",
        "visual_cache": indexes_dir / "visual_cache.json",
        "clip_cache": indexes_dir / "clip_visual_cache.json",
        "search_memory": indexes_dir / "search_memory.json",
        "page_previews": indexes_dir / "page_previews",
    }


def render_index_controls(runtime_root: Path) -> None:
    with st.expander("🗂️ Index"):
        st.caption(
            "Refresh after documents are added, deleted, renamed, or moved."
        )
        if is_azure_app_service():
            st.caption(f"Cloud runtime: `{runtime_root}`")
        local_clicked = st.button(
            "Refresh local index",
            use_container_width=True,
        )
        if local_clicked:
            refresh_local_index(runtime_root)

        if os.getenv("AZURE_BLOB_CONTAINER_URL"):
            blob_clicked = st.button(
                "Refresh Azure Blob index",
                use_container_width=True,
            )
            if blob_clicked:
                refresh_blob_index(runtime_root)


def refresh_local_index(runtime_root: Path) -> None:
    data_dir = PROJECT_ROOT / "data/raw"
    index_path = resolve_app_paths(runtime_root)["index"]
    try:
        records = collect_file_metadata(data_dir)
        save_json(records, index_path)
    except Exception as error:
        st.error(f"Could not refresh local index: {error}")
        return

    st.success(f"Refreshed local index with {len(records)} files.")


def refresh_blob_index(runtime_root: Path) -> None:
    container_url = os.getenv("AZURE_BLOB_CONTAINER_URL", "")
    index_path = resolve_app_paths(runtime_root)["index"]
    try:
        records = build_blob_index(index_path, container_url)
    except Exception as error:
        st.error(f"Could not refresh Azure Blob index: {error}")
        return

    st.success(f"Refreshed Azure Blob index with {len(records)} files.")


def ensure_index_exists(index_path: Path) -> None:
    if index_path.exists():
        return

    container_url = os.getenv("AZURE_BLOB_CONTAINER_URL", "").strip()
    if container_url:
        build_blob_index(index_path, container_url)
        return

    data_dir = PROJECT_ROOT / "data/raw"
    if data_dir.exists():
        records = collect_file_metadata(data_dir)
        save_json(records, index_path)
        return

    raise FileNotFoundError(
        "Index not found and no document source is configured. "
        "Set AZURE_BLOB_CONTAINER_URL or add documents under data/raw."
    )


def build_blob_index(index_path: Path, container_url: str) -> list[dict[str, object]]:
    if not container_url.strip():
        raise ValueError("AZURE_BLOB_CONTAINER_URL is empty.")

    records = collect_azure_blob_metadata(container_url)
    save_json(records, index_path)
    return records


def configure_page() -> None:
    st.set_page_config(
        page_title="Agentic Document Search",
        page_icon="🔎",
        layout="wide",
    )


def inject_ui_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ads-blue: #3b82f6;
            --ads-light-blue: #7fbbdd;
            --ads-soft-blue: #CBE3F1;
            --ads-orange: #f58b05;
            --ads-ink: #111827;
            --ads-muted: #6b7280;
            --ads-border: #d1d5db;
            --ads-form-font: 1.35rem;
            --ads-hero-copy-font: 1.5rem;
            --ads-query-bg: var(--ads-soft-blue);
            --ads-query-text: var(--ads-ink);
            --ads-query-placeholder: #4b5563;
        }

        div[data-testid="stMetric"] {
            display: none;
        }

        section[data-testid="stSidebar"],
        section[data-testid="stSidebar"] > div {
            background: var(--ads-soft-blue);
            color: var(--ads-ink) !important;
            color-scheme: light;
        }

        section[data-testid="stSidebar"] * {
            color: var(--ads-ink) !important;
        }

        .st-key-file_type_excel button,
        .st-key-file_type_word button,
        .st-key-file_type_powerpoint button,
        .st-key-file_type_pdf button {
            min-height: 3.7rem;
            border-radius: 8px;
            background: white;
            font-weight: 750;
            font-size: var(--ads-form-font);
            box-shadow: none !important;
        }

        .st-key-file_type_excel button {
            border: 1.5px solid #16833a !important;
            color: #16833a;
        }
        .st-key-file_type_word button {
            border: 1.5px solid #2563eb !important;
            color: #2563eb;
        }
        .st-key-file_type_powerpoint button {
            border: 1.5px solid #ea580c !important;
            color: #ea580c;
        }
        .st-key-file_type_pdf button {
            border: 1.5px solid #dc2626 !important;
            color: #dc2626;
        }

        .st-key-file_type_excel button p,
        .st-key-file_type_word button p,
        .st-key-file_type_powerpoint button p,
        .st-key-file_type_pdf button p,
        .st-key-search_button button p {
            font-size: var(--ads-form-font) !important;
            line-height: 1.2;
            white-space: nowrap;
        }

        .st-key-file_type_excel button:hover,
        .st-key-file_type_excel button:focus,
        .st-key-file_type_excel button:focus-visible,
        .st-key-file_type_excel button:active {
            border-color: #16833a !important;
            outline-color: #16833a !important;
            box-shadow: none !important;
        }

        .st-key-file_type_word button:hover,
        .st-key-file_type_word button:focus,
        .st-key-file_type_word button:focus-visible,
        .st-key-file_type_word button:active {
            border-color: #2563eb !important;
            outline-color: #2563eb !important;
            box-shadow: none !important;
        }

        .st-key-file_type_powerpoint button:hover,
        .st-key-file_type_powerpoint button:focus,
        .st-key-file_type_powerpoint button:focus-visible,
        .st-key-file_type_powerpoint button:active {
            border-color: #ea580c !important;
            outline-color: #ea580c !important;
            box-shadow: none !important;
        }

        .st-key-file_type_pdf button:hover,
        .st-key-file_type_pdf button:focus,
        .st-key-file_type_pdf button:focus-visible,
        .st-key-file_type_pdf button:active {
            border-color: #dc2626 !important;
            outline-color: #dc2626 !important;
            box-shadow: none !important;
        }

        .st-key-file_type_excel button[kind="primary"] {
            background: #16833a;
            color: white;
        }
        .st-key-file_type_word button[kind="primary"] {
            background: #2563eb;
            color: white;
        }
        .st-key-file_type_powerpoint button[kind="primary"] {
            background: #ea580c;
            color: white;
        }
        .st-key-file_type_pdf button[kind="primary"] {
            background: #dc2626;
            color: white;
        }

        .ads-query-label {
            color: var(--ads-ink);
            color: light-dark(var(--ads-ink), #f9fafb);
            font-size: var(--ads-form-font) !important;
            font-weight: 400;
            margin: 0;
            padding: 0;
        }

        .st-key-query_input {
            margin-top: -0.5rem;
            margin-bottom: -0.55rem;
        }

        .st-key-query_input textarea {
            min-height: 6.8rem;
            border-radius: 8px;
            border: 1.5px solid var(--ads-light-blue) !important;
            background: var(--ads-query-bg) !important;
            color: var(--ads-query-text) !important;
            font-size: var(--ads-form-font);
            resize: vertical;
        }

        .st-key-query_input textarea::placeholder {
            color: var(--ads-query-placeholder);
            opacity: 1;
        }

        .st-key-query_input div[data-baseweb="textarea"] {
            border-color: var(--ads-light-blue) !important;
            background: var(--ads-query-bg) !important;
        }

        .st-key-query_input textarea:focus,
        .st-key-query_input textarea:focus-visible,
        .st-key-query_input div[data-baseweb="textarea"]:focus-within {
            border-color: var(--ads-light-blue) !important;
            outline-color: var(--ads-light-blue) !important;
            box-shadow: 0 0 0 0.12rem rgba(127, 187, 221, 0.32) !important;
        }

        .st-key-search_button button {
            min-height: 3.25rem;
            border-radius: 8px;
            background: var(--ads-orange);
            border-color: var(--ads-orange);
            color: white;
            font-weight: 800;
            font-size: var(--ads-form-font);
        }

        .st-key-search_button button:hover,
        .st-key-search_button button:focus,
        .st-key-search_button button:focus-visible,
        .st-key-search_button button:active {
            background: var(--ads-light-blue) !important;
            border-color: var(--ads-light-blue) !important;
            color: white !important;
            box-shadow: none !important;
        }

        .ads-flow {
            display: flex;
            align-items: stretch;
            flex-wrap: wrap;
            gap: 0.55rem;
            overflow: visible;
            padding: 0.2rem 0 0.8rem;
            margin: 0.2rem 0 1rem;
        }

        .ads-step {
            position: relative;
            flex: 0 1 145px;
            padding: 0.62rem 1.2rem 0.62rem 0.85rem;
            min-height: 3.25rem;
            background: white;
            color: var(--ads-light-blue);
            clip-path: polygon(0 0, calc(100% - 18px) 0, 100% 50%, calc(100% - 18px) 100%, 0 100%, 12px 50%);
        }

        .ads-step::before {
            content: "";
            position: absolute;
            inset: 0;
            background: var(--ads-light-blue);
            clip-path: inherit;
            z-index: 0;
        }

        .ads-step::after {
            content: "";
            position: absolute;
            inset: 1.5px;
            background: white;
            clip-path: inherit;
            z-index: 0;
        }

        .ads-step > * {
            position: relative;
            z-index: 1;
        }

        .ads-step.done {
            background: var(--ads-light-blue);
            color: white;
        }

        .ads-step.active,
        .ads-step.current {
            background: var(--ads-orange);
            color: white;
        }

        .ads-step.done::before,
        .ads-step.done::after {
            background: var(--ads-light-blue);
        }

        .ads-step.current::before,
        .ads-step.current::after {
            background: var(--ads-orange);
        }

        .ads-step-name {
            font-size: 0.82rem;
            font-weight: 750;
            line-height: 1.15;
        }

        .ads-step-status {
            margin-top: 0.35rem;
            font-size: 0.72rem;
            color: currentColor;
            opacity: 0.78;
            text-transform: uppercase;
            letter-spacing: 0.02em;
        }

        .ads-score-wrap {
            display: flex;
            justify-content: flex-end;
            align-items: center;
            height: 100%;
        }

        .ads-score-ring {
            --score: 0;
            --score-color: #ef4444;
            width: 96px;
            aspect-ratio: 1;
            border-radius: 50%;
            background:
                radial-gradient(closest-side, white 66%, transparent 68%),
                conic-gradient(var(--score-color) calc(var(--score) * 1%), #e5e7eb 0);
            display: grid;
            place-items: center;
            color: var(--ads-ink);
            font-weight: 950;
            font-size: 1.65rem;
        }

        .ads-reasons {
            margin: 0.2rem 0 0.75rem;
            padding-left: 1.1rem;
        }
        
        .ads-hero {
            padding: 0.4rem 0 0.25rem;
        }

        .ads-hero h1 {
            color: var(--ads-ink);
            color: light-dark(var(--ads-ink), #f9fafb);
            font-size: 3.2rem;
            line-height: 1.05;
            margin-bottom: 0.45rem;
        }

        .ads-hero-main {
            color: var(--ads-orange);
            font-size: var(--ads-hero-copy-font) !important;
            font-weight: 800;
            margin: 0;
            line-height: 1.22;
        }

        .ads-hero-sub {
            color: var(--ads-muted);
            color: light-dark(var(--ads-muted), #d1d5db);
            font-size: var(--ads-hero-copy-font) !important;
            margin: 0;
            line-height: 1.22;
        }
        
        .ads-file-type-spacer {
            height: 0;
        }

        .ads-file-caption {
            color: var(--ads-muted);
            color: light-dark(var(--ads-muted), #d1d5db);
            font-size: var(--ads-form-font) !important;
            font-weight: 400;
            margin: 0.15rem 0 0.35rem;
            line-height: 1.25;
        }

        .ads-example-spacer {
            height: 2cm;
        }

        .ads-inline-warning {
            max-width: 38rem;
            margin: 0.8rem auto 0;
            padding: 0.8rem 1rem;
            border: 1.5px solid var(--ads-orange);
            border-radius: 8px;
            background: #fff7ed;
            color: #9a3412;
            font-size: 1rem;
            font-weight: 600;
            text-align: center;
        }

        .ads-reasons li {
            margin: 0.2rem 0;
        }

        @media (prefers-color-scheme: dark) {
            .ads-hero h1 {
                color: #f9fafb !important;
            }

            .ads-query-label {
                color: #f9fafb !important;
            }

            .ads-hero-sub,
            .ads-file-caption {
                color: #d1d5db !important;
            }

            .st-key-query_input textarea,
            .st-key-query_input div[data-baseweb="textarea"] {
                background: var(--ads-query-bg) !important;
                color: var(--ads-query-text) !important;
            }
        }

        html[data-theme="dark"] .ads-hero h1,
        body[data-theme="dark"] .ads-hero h1,
        [data-testid="stApp"][data-theme="dark"] .ads-hero h1,
        [data-color-mode="dark"] .ads-hero h1 {
            color: #f9fafb !important;
        }

        html[data-theme="dark"] .ads-query-label,
        body[data-theme="dark"] .ads-query-label,
        [data-testid="stApp"][data-theme="dark"] .ads-query-label,
        [data-color-mode="dark"] .ads-query-label {
            color: #f9fafb !important;
        }

        html[data-theme="dark"] .ads-hero-sub,
        html[data-theme="dark"] .ads-file-caption,
        body[data-theme="dark"] .ads-hero-sub,
        body[data-theme="dark"] .ads-file-caption,
        [data-testid="stApp"][data-theme="dark"] .ads-hero-sub,
        [data-testid="stApp"][data-theme="dark"] .ads-file-caption,
        [data-color-mode="dark"] .ads-hero-sub,
        [data-color-mode="dark"] .ads-file-caption {
            color: #d1d5db !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def format_search_mode(value: str) -> str:
    return {
        "auto": "🎯 Auto Search",
        "instant": "⚡ Filename Search",
        "reasoning": "📦 Content Search",
        "visual": "👁️ Visual Search",
    }[value]


def mode_summary(value: str) -> str:
    return {
        "auto": "Automatically choose the best search method.",
        "instant": "Search using filenames, folders, and file details.",
        "reasoning": "Open documents and match their contents.",
        "visual": "Match documents by content, layout, charts, and images.",
    }[value]


def preset_settings(value: str) -> dict[str, object]:
    presets = {
        "instant": {
            "mode": "llm",
            "content_mode": "never",
            "translator_mode": "never",
            "azure_ai_search_mode": "never",
            "visual_mode": "never",
            "visual_prefilter": "none",
            "candidate_pool_size": 6,
            "max_inspected_files": 6,
            "max_pages_per_file": 200,
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
            "max_pages_per_file": 200,
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
            "max_pages_per_file": 200,
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
            "max_pages_per_file": 200,
            "max_visual_files": 3,
            "max_visual_pages_per_file": 5,
            "max_clip_pages": 10,
        },
    }
    return presets[value]


def show_empty_state() -> None:
    st.markdown(
        """
        <div class="ads-example-spacer"></div>

        **Example queries**

        - ⚡ *Apache関連だった気がする、Engineering配下にあった年次レポート*
        - 📦 *A document discussing open-source ecosystem trends and community growth in 2025.*
        - 👁️ *A telecom-related presentation with revenue graphs and a distinctive magenta corporate design.*
        """,
        unsafe_allow_html=True,
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
    st.subheader("Agent Flow")
    render_flow_html(response.steps, current_step=None)

    with st.expander("Step details"):
        for step in response.steps:
            display_name = display_step_name(str(step.name))
            st.markdown(f"**{display_name}** · `{step.status}`")
            st.caption(step.detail)


def render_progress_flow(
    placeholder: object,
    steps: list[object],
    current_step: str | None,
) -> None:
    with placeholder.container():
        st.subheader("Agent Flow")
        render_flow_html(steps, current_step=current_step)


def render_flow_html(steps: list[object], current_step: str | None) -> None:
    visible_steps: dict[str, str] = {}
    for step in steps:
        step_key = normalize_flow_step_name(str(step.name))
        if not step_key:
            continue
        status = str(step.status).lower()
        if status in {"done", "partial", "fallback", "deferred"}:
            visible_steps[step_key] = status

    active = normalize_flow_step_name(current_step) if current_step else None
    cards = []
    for step_key, label in FLOW_STEPS:
        if step_key == active:
            state_class = "current"
            state_label = "current"
        elif step_key in visible_steps:
            state_class = "done" if visible_steps[step_key] != "deferred" else "current"
            state_label = visible_steps[step_key]
        else:
            continue
        cards.append(
            "<div class='ads-step "
            + state_class
            + "'><div class='ads-step-name'>"
            + escape(label)
            + "</div><div class='ads-step-status'>"
            + escape(state_label)
            + "</div></div>"
        )
    render_html("<div class='ads-flow'>" + "".join(cards) + "</div>")


def normalize_flow_step_name(name: str | None) -> str | None:
    if not name:
        return None
    return FLOW_ALIASES.get(name, name)


def display_step_name(name: str) -> str:
    normalized = normalize_flow_step_name(name) or name
    return FLOW_LABELS.get(normalized, name)


def render_html(html: str) -> None:
    if hasattr(st, "html"):
        st.html(html)
    else:
        st.markdown(html, unsafe_allow_html=True)


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
                show_score_ring(result.score)

            st.markdown("**Why this matched**")
            reasons = clean_display_reasons(result.reasons)
            if reasons:
                st.markdown(
                    "<ul class='ads-reasons'>"
                    + "".join(f"<li>{escape(reason)}</li>" for reason in reasons)
                    + "</ul>",
                    unsafe_allow_html=True,
                )
            else:
                st.caption("The file stayed in the candidate set after metadata and content checks.")

            show_page_preview(result)


def show_score_ring(score: float) -> None:
    bounded = max(0.0, min(float(score), 100.0))
    color = score_color(bounded)
    st.markdown(
        f"""
        <div class="ads-score-wrap">
            <div>
                <div
                    class="ads-score-ring"
                    style="--score:{bounded:.1f}; --score-color:{color};"
                >
                    {bounded:.0f}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def score_color(score: float) -> str:
    score = max(0.0, min(score, 100.0))
    if score < 50:
        ratio = score / 50
        return interpolate_hex("#ef4444", "#facc15", ratio)
    ratio = (score - 50) / 50
    return interpolate_hex("#facc15", "#16a34a", ratio)


def interpolate_hex(start: str, end: str, ratio: float) -> str:
    ratio = max(0.0, min(ratio, 1.0))
    start_rgb = hex_to_rgb(start)
    end_rgb = hex_to_rgb(end)
    mixed = [
        round(start_value + (end_value - start_value) * ratio)
        for start_value, end_value in zip(start_rgb, end_rgb)
    ]
    return "#" + "".join(f"{value:02x}" for value in mixed)


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return (
        int(value[0:2], 16),
        int(value[2:4], 16),
        int(value[4:6], 16),
    )


def clean_display_reasons(reasons: list[str]) -> list[str]:
    if any(has_cjk(reason) for reason in reasons):
        cleaned = []
        for reason in reasons:
            compact = trim_reason(reason)
            if compact and not has_cjk(compact):
                compact = translate_reason_with_template(compact)
            if compact and compact not in cleaned:
                cleaned.append(compact)
        return cleaned[:5]

    buckets: dict[str, list[str]] = {
        "visual": [],
        "content": [],
        "metadata": [],
        "memory": [],
        "other": [],
    }
    for reason in reasons:
        compact = compact_reason(reason)
        if not compact:
            continue
        category = reason_category(compact)
        if compact not in buckets[category]:
            buckets[category].append(compact)

    ordered = (
        buckets["visual"][:2]
        + buckets["content"][:2]
        + buckets["metadata"][:2]
        + buckets["memory"][:1]
        + buckets["other"][:1]
    )
    return ordered[:5]


def reason_category(reason: str) -> str:
    lower = reason.lower()
    if lower.startswith("visual"):
        return "visual"
    if lower.startswith("content") or lower.startswith("text evidence"):
        return "content"
    if lower.startswith("search memory"):
        return "memory"
    if (
        "filename" in lower
        or "folder" in lower
        or "path" in lower
        or "metadata" in lower
        or lower.startswith("matched")
        or lower.startswith("fuzzy")
    ):
        return "metadata"
    return "other"


def compact_reason(reason: str) -> str:
    reason = reason.strip()
    if not reason:
        return ""

    lower = reason.lower()
    technical_markers = [
        "cache hit",
        "embedding cache",
        "similarity",
        "optional dependencies",
        "run: .venv",
        "could not load",
        "could not create",
    ]
    if any(marker in lower for marker in technical_markers):
        return ""

    if reason.startswith("CLIP selected visual pages:") or reason.startswith("Visual page candidates:"):
        pages = ", ".join(re.findall(r"\d+", reason))
        return f"Visual page candidates: {pages}" if pages else "Visual pages looked relevant"

    if reason.startswith("CLIP observation"):
        return ""

    if lower.startswith("visual analysis matched terms:") or lower.startswith("visual analysis matched:"):
        terms = reason.split(":", 1)[1].strip()
        return f"Visual analysis matched: {terms}" if terms else "Visual analysis supported the match"

    if lower.startswith("visual observation "):
        return trim_reason(reason.replace("visual observation ", "", 1))

    if reason.startswith("content text matched:"):
        terms = reason.split(":", 1)[1].strip()
        return f"Content matched: {terms}" if terms else "Content matched the query"

    if reason.startswith("content snippet "):
        return trim_reason(reason.replace("content snippet ", "Text evidence: ", 1))

    if reason.startswith("search memory:"):
        return "Search memory recalled similar past searches"

    metadata_match = re.match(r"matched '([^']+)' in (.+)", reason)
    if metadata_match:
        term, field = metadata_match.groups()
        if "folder" in field or "path" in field:
            return f"Folder/path matched: {term}"
        if "filename" in field or "title" in field:
            return f"Filename/title matched: {term}"
        return f"Metadata matched: {term}"

    fuzzy_match = re.match(r"fuzzy matched '([^']+)' to '([^']+)' in (.+)", reason)
    if fuzzy_match:
        query_term, matched_term, field = fuzzy_match.groups()
        if "filename" in field or "title" in field:
            return f"Filename/title fuzzy-matched {query_term} to {matched_term}"
        if "folder" in field or "path" in field:
            return f"Folder/path fuzzy-matched {query_term} to {matched_term}"
        return f"Metadata fuzzy-matched {query_term} to {matched_term}"

    if reason.startswith("local OCR fallback note:"):
        return ""

    return trim_reason(reason)


def trim_reason(reason: str, max_chars: int = 170) -> str:
    reason = " ".join(reason.split())
    if len(reason) <= max_chars:
        return reason
    return reason[: max_chars - 1].rstrip() + "…"


def show_page_preview(result: object) -> None:
    record = result.record
    if str(record.get("extension", "")).lower() != ".pdf":
        return

    preview_pages = extract_preview_pages(result.reasons)
    visual_panel = has_visual_evidence(result.reasons)

    try:
        image_paths = render_preview_pages(record, preview_pages[:4])
    except Exception as error:
        st.warning(f"Could not render preview: {error}")
        return

    if not image_paths:
        return

    title = "Visual investigation" if visual_panel else "Preview"
    st.markdown(f"**{title}**")
    columns = st.columns(min(len(image_paths), 4))
    for column, (image_path, page_number) in zip(columns, image_paths):
        with column:
            st.image(
                str(image_path),
                caption=(
                    f"Inspected page {page_number}"
                    if visual_panel
                    else f"Page {page_number}"
                ),
                use_container_width=True,
            )


def has_visual_evidence(reasons: list[str]) -> bool:
    return any(
        marker in reason
        for reason in reasons
        for marker in (
            "Visual page candidates:",
            "Visual analysis matched:",
            "visual observation",
            "visual skim",
        )
    )


def extract_preview_pages(reasons: list[str]) -> list[int]:
    pages: list[int] = []
    for reason in reasons:
        if (
            "CLIP selected visual pages:" in reason
            or "Visual page candidates:" in reason
            or "local visual skim selected pages:" in reason
            or ("視覚" in reason and "ページ" in reason)
        ):
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
    preview_dir = resolve_app_paths(resolve_runtime_root())["page_previews"]
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
