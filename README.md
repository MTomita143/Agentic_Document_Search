# Visual Memory Copilot

Find the slide or document you vaguely remember from its name, folder, content, graph, layout, or visual appearance.

## Concept

The search flow should feel like how a person looks for a file:

```text
Name-based guess
→ Look inside the candidate files
→ Check visual information
→ Return the best match with reasons
```

## Current Phase

### Phase 1: File-level metadata search

The first implementation only creates a lightweight metadata index from files.

It does **not**:
- read document content
- generate summaries
- create embeddings
- render slides/pages as images
- use LLM reasoning

It only saves lightweight metadata:

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

## Why metadata first?

The goal is to avoid heavy preprocessing at the beginning.

Metadata is cheap and useful:
- file name
- title from file name
- folder name
- parent folder
- grandparent folder
- extension

This supports the first step:

```text
"Maybe it was in the revenue folder"
"Maybe the title had Q4"
"Maybe it was a PowerPoint"
```

## Project Structure

```text
visual-memory-copilot/
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

Example:

```text
data/raw/
├── CompanyA/
│   └── Reports/
│       └── Q4_Revenue_Update.pptx
└── Strategy/
    └── Market_Expansion.pdf
```

Run:

```bash
python src/ingest.py --data-dir data/raw --output indexes/files_index.json
```

## Roadmap / TODO

### MVP v0
- [x] Create repository structure
- [x] Create metadata ingestion script
- [x] Save file-level metadata as JSON
- [ ] Add simple keyword/fuzzy search over metadata
- [ ] Return top-k candidate files with reasons

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

## Design Principle

Do not process everything in advance.

Start light:

```text
metadata first
→ content on demand
→ visual information on demand
→ cache only when useful
```
