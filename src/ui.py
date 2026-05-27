"""
Streamlit prototype for Agentic Document Search.

Run locally:
    .venv/bin/streamlit run src/ui.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

from agent import run_agent
from search import DEFAULT_INDEX_PATH, load_environment


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    load_environment()
    configure_page()

    st.title("Agentic Document Search")
    st.caption("Find documents the way you remember them.")

    query = st.text_input(
        "What do you remember?",
        placeholder="Example: I remember a slide with a blue graph",
    )

    with st.sidebar:
        st.header("Search Settings")

        search_mode = st.radio(
            "Search mode",
            options=["instant", "reasoning", "visual"],
            format_func=format_search_mode,
            index=0,
            help="Choose how much the agent should inspect before answering.",
        )
        settings = preset_settings(search_mode)
        st.caption(mode_summary(search_mode))

        with st.expander("⚙️ Customize mode"):
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
        visual_prefilter = "clip" if use_clip else "none"
        visual_mode = "azure" if use_vision else "never"

    show_cost_notice(
        mode,
        translator_mode,
        visual_mode,
        visual_prefilter,
        max_visual_files,
        max_visual_pages_per_file,
        max_clip_pages,
    )

    search_clicked = st.button("Search", type="primary")

    if not query:
        if search_clicked:
            st.warning("Enter a search memory first.")
        show_empty_state()
        return

    if not search_clicked:
        return

    index_path = PROJECT_ROOT / DEFAULT_INDEX_PATH
    content_cache_path = PROJECT_ROOT / "indexes/content_cache.json"
    visual_cache_path = PROJECT_ROOT / "indexes/visual_cache.json"
    clip_cache_path = PROJECT_ROOT / "indexes/clip_visual_cache.json"

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
            )
        except Exception as error:
            st.error(str(error))
            return

    show_agent_steps(response)
    show_results(response)


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
    }[value]


def mode_summary(value: str) -> str:
    return {
        "instant": "Metadata plus local text inspection. No Azure AI calls.",
        "reasoning": "Instant search plus Azure OpenAI reranking.",
        "visual": "Reasoning search plus CLIP prefilter and Azure visual inspection.",
    }[value]


def preset_settings(value: str) -> dict[str, object]:
    presets = {
        "instant": {
            "mode": "local",
            "content_mode": "auto",
            "translator_mode": "never",
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
            "visual_mode": "azure",
            "visual_prefilter": "clip",
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
    mode: str,
    translator_mode: str,
    visual_mode: str,
    visual_prefilter: str,
    max_visual_files: int,
    max_visual_pages_per_file: int,
    max_clip_pages: int,
) -> None:
    if mode == "llm":
        st.warning("Azure OpenAI reranking is enabled. This may use paid tokens.")

    if translator_mode == "auto":
        st.info("Azure Translator query expansion is enabled for Japanese prompts.")

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
