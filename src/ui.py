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

        mode = st.radio(
            "Metadata ranking",
            options=["local", "llm"],
            format_func=lambda value: {
                "local": "Local metadata search",
                "llm": "Azure OpenAI rerank",
            }[value],
            help="Azure OpenAI uses your Foundry deployment and may cost money.",
        )

        content_mode = st.selectbox(
            "Text inspection",
            options=["auto", "always", "never"],
            index=0,
            help="Inspects extracted PDF text from top candidate files only.",
        )

        visual_mode = st.selectbox(
            "Visual inspection",
            options=["auto", "azure", "never"],
            index=0,
            help="Azure Vision sends rendered PDF pages to Azure AI Vision and may cost money.",
        )

        st.divider()
        st.header("Limits")
        top_k = st.slider("Results", min_value=1, max_value=6, value=3)
        candidate_pool_size = st.slider(
            "LLM candidate pool",
            min_value=1,
            max_value=10,
            value=6,
            help="Only used for Azure OpenAI reranking.",
        )
        max_inspected_files = st.slider("Text files to inspect", 1, 6, 3)
        max_pages_per_file = st.slider("Text pages per file", 1, 40, 20)
        max_visual_files = st.slider("Visual files to inspect", 1, 4, 2)
        max_visual_pages_per_file = st.slider("Visual pages per file", 1, 6, 3)

    show_cost_notice(mode, visual_mode, max_visual_files, max_visual_pages_per_file)

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

    with st.spinner("Running agentic search..."):
        try:
            response = run_agent(
                query=query,
                index_path=index_path,
                mode=mode,
                top_k=top_k,
                candidate_pool_size=candidate_pool_size,
                content_mode=content_mode,
                max_inspected_files=max_inspected_files,
                max_pages_per_file=max_pages_per_file,
                max_chars_per_file=30000,
                content_cache_path=content_cache_path,
                visual_mode=visual_mode,
                max_visual_files=max_visual_files,
                max_visual_pages_per_file=max_visual_pages_per_file,
                visual_cache_path=visual_cache_path,
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


def show_empty_state() -> None:
    st.info("Try a vague memory like: “Find the report about open source.”")
    st.markdown(
        """
        **Good demo queries**

        - `Find the investor presentation deck`
        - `Find a report about open source`
        - `Find the document that talks about telecom`
        - `I remember a slide with a blue graph`
        """
    )


def show_cost_notice(
    mode: str,
    visual_mode: str,
    max_visual_files: int,
    max_visual_pages_per_file: int,
) -> None:
    if mode == "llm":
        st.warning("Azure OpenAI reranking is enabled. This may use paid tokens.")

    if visual_mode == "azure":
        max_calls = max_visual_files * max_visual_pages_per_file
        st.warning(
            f"Azure Vision is enabled. This run can make up to {max_calls} image-analysis calls."
        )
    elif visual_mode == "auto":
        max_calls = max_visual_files * max_visual_pages_per_file
        st.info(
            "Azure Vision may run only when the query has visual clues. "
            f"If it runs, the current maximum is {max_calls} image-analysis calls."
        )


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
