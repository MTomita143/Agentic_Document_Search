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

Run the visible agent flow:
```bash
python3 src/agent.py --query "find the investor presentation deck" --mode local
```

Run the Streamlit prototype:
```bash
.venv/bin/streamlit run src/ui.py
```

Optional: install the local CLIP visual prefilter:
```bash
.venv/bin/python -m pip install -r requirements-clip.txt
```

The agent flow currently shows:

```text
understand query
→ search English/Japanese/mixed metadata
→ inspect extracted English/Japanese/mixed text from top candidate documents when useful
→ use local OCR fallback for scanned PDF pages when extracted text is too thin
→ prefilter rendered visual pages locally with CLIP
→ decide whether visual inspection is needed next
→ return ranked matches with reasons
```

Force local content inspection for the top candidates:
```bash
.venv/bin/python src/agent.py --query "find a report about open source" --mode local --content-mode always
```

Japanese and mixed-language queries are supported in the local layer when the
query and candidate documents share Japanese or English terms:

```bash
.venv/bin/python src/agent.py --query "売上分析のレポートを探して" --mode local
.venv/bin/python src/agent.py --query "青いグラフがある資料" --mode local
```

The local tokenizer uses Unicode normalization plus Japanese Kanji/Katakana
term matching, so filenames and extracted text can be Japanese, English, or
mixed. If the query is Japanese but the target document only contains English
terms, use Azure OpenAI reranking or add Azure Translator query expansion later
as a paid semantic bridge.

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

AZURE_VISION_ENDPOINT="https://YOUR-VISION-RESOURCE.cognitiveservices.azure.com/"
AZURE_VISION_KEY="YOUR-VISION-KEY"
AZURE_VISION_API_VERSION="2024-02-01"
AZURE_VISION_FEATURES="caption,denseCaptions,tags,read,objects"

ADS_OCR_MIN_CHARS="120"
ADS_OCR_LANGS="eng"
```

Run:
```bash
.venv/bin/python src/search.py --query "the board slide deck" --mode llm --top-k 3
```

Run the agent flow with Azure OpenAI metadata reranking:
```bash
.venv/bin/python src/agent.py --query "the board slide deck" --mode llm --top-k 3
```

The Azure OpenAI reranker receives only candidate file metadata. The agent can
then inspect extracted PDF/PPTX/DOCX/XLSX text locally for top candidates when useful. Visual
inspection renders only selected top-candidate PDF pages and sends those images
to Azure Vision when `--visual-mode azure` is used or when `--visual-mode auto`
sees visual clues.

If a candidate PDF has very little extractable text, the content layer can use
local Tesseract OCR as a fallback. This is intentionally local and free of Azure
OCR calls. The fallback is best for scanned PDFs. PPTX/DOCX image-only OCR needs
a separate slide/page rendering strategy, so it is left as a later architecture
choice. Set `ADS_OCR_LANGS` for installed Tesseract languages; for Japanese
scanned documents, the local machine or deployment image needs Japanese
Tesseract language data.

Run Azure Vision visual inspection:
```bash
.venv/bin/python src/agent.py \
  --query "I remember a slide with a blue graph" \
  --mode local \
  --visual-prefilter clip \
  --visual-mode azure \
  --max-visual-files 2 \
  --max-visual-pages-per-file 6 \
  --max-clip-pages 3
```

With `--visual-prefilter clip`, the agent ranks rendered pages locally first and
sends only the top CLIP-selected pages to Azure Vision. If CLIP is selected but
not installed, Azure Vision is skipped to avoid sending unfiltered pages to a
billable verifier.

## Azure App Service

For the hackathon deployment, Streamlit is the web interface and Azure App
Service is the application execution platform.

Use this startup command on App Service:

```bash
python -m streamlit run src/ui.py --server.port 8000 --server.address 0.0.0.0
```

Set the Azure OpenAI and Azure Vision values as App Service configuration
environment variables. Do not upload a local `.env` file.

The base deployment uses `requirements.txt`. CLIP is intentionally kept in
`requirements-clip.txt` because it pulls in a heavier local ML stack. Add it to
the deployment only if the App Service plan can handle the extra install size
and startup time.

## Roadmap / TODO

### MVP v0
- [x] Create repository structure
- [x] Create metadata ingestion script
- [x] Save file-level metadata as JSON
- [x] Add simple keyword/fuzzy search over metadata
- [x] Return top-k candidate files with reasons
- [x] Add optional Azure OpenAI metadata reranking

### MVP v1
- [x] Read content only from candidate files
- [x] Extract text from PDF
- [x] Extract text from PPTX
- [x] Extract text from DOCX
- [x] Extract text from XLSX/XLSM
- [x] Add local OCR fallback for scanned PDF pages with weak extracted text
- [x] Support Japanese and Japanese/English mixed metadata and content search
- [x] Search page/slide/chunk/sheet-level text only after file-level filtering

### MVP v2
- [x] Render selected PDF pages as images for visual inspection
- [x] Add optional local CLIP page prefilter
- [x] Wire Azure Vision analysis for selected candidate pages
- [ ] Add robust visual labels such as graph, table, blue theme, layout type
- [ ] Search using visual memory queries

### MVP v3
- [x] Add Azure OpenAI metadata reranking
- [x] Generate user-facing explanation:
  - why this file matched
  - which metadata/content/visual clues were used
- [ ] Generate a polished final natural-language answer with an LLM

### Later idea: Lazy cache
- [ ] Cache extracted text only after a file is opened once
- [ ] Cache generated summaries only for frequently searched files
- [ ] Cache visual labels only for pages/slides that were actually inspected
- [ ] Avoid indexing everything heavily upfront
