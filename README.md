# Agentic Document Search
🌟Find Documents the Way You Remember Them

Instead of relying on exact keywords, this AI agent searches documents the way humans do — through vague memory, visual cues, and iterative exploration.

## Concept

A human-like search process:

```text
Vague recall (name, context, or topic)
→ Narrow down candidates
→ Look inside documents
→ Use visual signals (charts, layouts, slides)
→ Refine the search iteratively
→ Return the most relevant results with reasoning
```

## Project Structure

```text
./
├── data/
│   └── raw/
│       └── put your pdf/pptx/docx files here
├── indexes/
│   └── files_index.json
├── src/
│   └── ingest.py
├── .gitignore
└── README.md
```

## Usage

Put documents under:
```text
data/raw/
```

Run:
```bash
python3 src/ingest.py --data-dir data/raw --output indexes/files_index.json
```

Search metadata locally:
```bash
python3 src/search.py --query "investor presentation" --mode local --top-k 3
```

Search metadata with Azure OpenAI reranking:
```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Create a local `.env` file:
```bash
AZURE_OPENAI_ENDPOINT="https://YOUR-RESOURCE.cognitiveservices.azure.com/"
AZURE_OPENAI_API_KEY="YOUR-KEY"
AZURE_OPENAI_DEPLOYMENT="gpt-5-mini"
AZURE_OPENAI_API_VERSION="2024-12-01-preview"
```

Run:
```bash
.venv/bin/python src/search.py --query "the board slide deck" --mode llm --top-k 3
```

The LLM mode receives only candidate file metadata. It does not open document
contents or inspect visual information.

## Roadmap / TODO

### MVP v0
- [x] Create repository structure
- [x] Create metadata ingestion script
- [x] Save file-level metadata as JSON
- [x] Add simple keyword/fuzzy search over metadata
- [x] Return top-k candidate files with reasons
- [x] Add optional Azure OpenAI metadata reranking

### MVP v1
- [ ] Read content only from candidate files
- [ ] Extract text from PDF
- [ ] Extract text from PPTX
- [ ] Search page/slide-level text only after file-level filtering

### MVP v2
- [ ] Render selected pages/slides as images
- [ ] Add visual labels such as graph, table, blue theme, layout type
- [ ] Search using visual memory queries

### MVP v3
- [ ] Add LLM reranking
- [ ] Generate user-facing explanation:
  - why this file matched
  - which metadata/content/visual clues were used

### Later idea: Lazy cache
- [ ] Cache extracted text only after a file is opened once
- [ ] Cache generated summaries only for frequently searched files
- [ ] Cache visual labels only for pages/slides that were actually inspected
- [ ] Avoid indexing everything heavily upfront
