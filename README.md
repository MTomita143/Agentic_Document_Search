# Agentic Document Search

**Find Documents the Way You Remember Them**

Agentic Document Search is a hackathon MVP for finding documents from vague
memory, not only exact filenames or keywords.

People often remember work documents like this:

```text
"the PDF about telecom with graphs"
"the investor presentation deck"
"the report that mentioned Apache and charts"
"青いグラフがあった資料"
```

This project turns that kind of memory into an agentic search loop:

```text
Vague recall
→ lightweight metadata narrowing
→ candidate-only text inspection
→ optional visual page selection
→ optional Azure Vision verification
→ ranked results with reasons
```

The core principle is:

```text
Cheap first, expensive only when useful.
```

The agent does not deeply process every document upfront. It starts with
filenames, folder paths, file type, and previous search memory, then opens only
the most likely candidate files.

## Current Architecture

```text
User
  ↓
Streamlit UI
  ↓
Agent controller
  ↓
Semantic Kernel auto mode
  ↓
Azure Translator query expansion
  ↓
Fast Azure OpenAI metadata reasoning
  ↓
Search memory + optional Azure AI Search
  ↓
Candidate-only text extraction
  ↓
Deep Azure OpenAI content reasoning
  ↓
Hybrid visual page selection
  ↓
Optional CLIP / GPU VM worker
  ↓
Optional Azure Vision verification
  ↓
Explained result
```

Azure deployment shape:

```text
Azure App Service
  Streamlit UI + agent controller

Azure Blob Storage
  Cloud document source for demo/deployment

Azure VM / Azure GPU VM
  Optional home for heavier local workers; current worker runs CLIP page ranking

Microsoft Foundry / Azure OpenAI
  Query planning, metadata reranking, content reranking, Japanese reason rewriting

Azure AI Vision
  Visual verification for selected rendered pages

Azure AI Translator
  Japanese/English query expansion for mixed-language search

Azure AI Search
  Optional candidate retrieval tool, not the main product logic

Semantic Kernel
  Auto mode orchestration: chooses which tools to use for the query
```

Future enterprise source:

```text
OneDrive / SharePoint / Microsoft Graph
  Real company document source-of-truth
```

The product is not "upload every document into a new AI database." The product
is a policy-aware document search agent that can investigate approved company
storage with bounded computation.

## What Makes It Agentic

Normal search usually does this:

```text
query → retrieve → return results
```

This project does this:

```text
query
→ decide strategy
→ search metadata
→ remember previous investigations
→ inspect only candidate files
→ inspect visual pages only when the query needs it
→ compare evidence
→ explain why each file matched
```

Azure AI Search can be used as one retrieval tool, but the agentic behavior is
the orchestration around it: deciding when to use metadata, text, memory,
translation, CLIP, Vision, and deep reranking.

## Search Modes

The Streamlit UI exposes four modes:

```text
Auto Search
  Semantic Kernel chooses the search strategy.

Filename Search
  Filenames, folders, and file details. No file content inspection.

Content Search
  Metadata search + Azure OpenAI reranking + candidate text inspection.

Visual Search
  Content Search + hybrid visual page selection + optional CLIP/Azure Vision.
```

File type buttons can narrow the candidate set before search:

```text
Excel | Word | PowerPoint | PDF
```

Visual inspection is useful for PDF/slide-like files. Word and Excel searches
lean on text extraction and LLM reasoning instead.

## Agent Flow

The current agent can run these steps:

```text
1. File type filter
2. Azure Translator query expansion
3. Query understanding
4. Semantic Kernel auto-mode planning
5. Metadata search
6. Azure OpenAI metadata rerank
7. Search memory recall
8. Optional Azure AI Search retrieval
9. Candidate text extraction
10. Local OCR fallback for scanned PDF pages
11. Deep Azure OpenAI content rerank
12. Local visual page skim
13. Optional CLIP page prefilter
14. Optional Azure Vision page verification
15. Japanese/English reason rendering
```

The UI shows completed/current steps during search so the demo reads like an
investigation, not a black-box lookup.

## Language Strategy

The app supports Japanese, English, and mixed Japanese/English documents.

The internal search path prefers English when using LLM/Vision/CLIP because
those models tend to be more stable on English visual and semantic cues. When
the user prompt is Japanese:

```text
Japanese prompt
→ Azure Translator expands the query
→ search uses Japanese + English clues
→ CLIP uses the English visual query when available
→ Azure OpenAI deep deployment rewrites reasons in Japanese
```

The UI stays in English, but `Why this matched` is rendered in Japanese for
Japanese prompts. Filenames, company names, paths, and evidence terms such as
`FIGURE 1`, `Apache`, `chart`, `CLIP`, and `Azure Vision` are preserved.

## Metadata Design

The file index stores only canonical fields:

```json
{
  "file_id": "file_xxxxx",
  "source": "local",
  "uri": "company-drive/Reports/file.pdf",
  "size_bytes": 1234567,
  "modified_time": "2026-05-27T12:00:00+00:00"
}
```

Filename, extension, folder names, type labels, display size, and local paths
are derived at load time. This keeps the index portable when moving from a
local folder to Azure Blob Storage.

File size is kept for display and cache invalidation, not as a primary ranking
reason. It is usually a weak signal for human-memory search.

Refresh the index after files are added, deleted, renamed, or moved.

## Data Efficiency

The project uses two memory layers:

```text
Lazy file cache
  extracted text, OCR output, rendered page info, CLIP embeddings, Vision output

Search memory
  previous query, recalled documents, timestamp, and short reason summary
```

This is intentionally different from permanent visual labeling. A human does
not label every page in advance; they remember what they already investigated
and start from similar clues next time.

## Project Structure

```text
.
├── data/
│   └── raw/
│       └── company-drive/
├── indexes/
│   ├── files_index.json
│   ├── content_cache.json
│   ├── visual_cache.json
│   ├── clip_cache.json
│   └── search_memory.json
├── src/
│   ├── agent.py
│   ├── ingest.py
│   ├── search.py
│   ├── content.py
│   ├── visual.py
│   ├── visual_profile.py
│   ├── clip_prefilter.py
│   ├── visual_worker_client.py
│   ├── semantic_kernel_auto.py
│   ├── translator.py
│   ├── reason_localizer.py
│   └── ui.py
├── vm_worker/
│   └── app.py
├── requirements.txt
├── requirements-clip.txt
├── requirements-semantic-kernel.txt
└── .env.example
```

## Local Setup

Create a virtual environment and install the base app:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

Optional Semantic Kernel auto mode:

```bash
.venv/bin/python -m pip install -r requirements-semantic-kernel.txt
```

Optional local CLIP:

```bash
.venv/bin/python -m pip install -r requirements-clip.txt
```

CLIP is optional because it is heavier than the base Streamlit app. If you use
it locally, model files are cached under `indexes/model_cache/huggingface` by
default.

## Environment

Start from `.env.example` and create a local `.env` file.

Core Azure settings:

```bash
AZURE_OPENAI_ENDPOINT="https://YOUR-RESOURCE.cognitiveservices.azure.com/"
AZURE_OPENAI_API_KEY="YOUR-KEY"
AZURE_OPENAI_FAST_DEPLOYMENT="gpt-5.4-mini"
AZURE_OPENAI_DEEP_DEPLOYMENT="gpt-5.4"
AZURE_OPENAI_API_VERSION="2024-12-01-preview"

AZURE_BLOB_CONTAINER_URL="https://YOUR-STORAGE.blob.core.windows.net/YOUR-CONTAINER?YOUR-SAS"
AZURE_TRANSLATOR_REGION="YOUR-AZURE-AI-RESOURCE-REGION"
```

The deployment names must match the names in Azure AI Foundry exactly. The
deployment name can differ from the model name.

Vision and Translator reuse the Foundry endpoint/key by default when they live
in the same Azure AI resource:

```bash
AZURE_VISION_API_VERSION="2024-02-01"
AZURE_VISION_FEATURES="caption,denseCaptions,tags,read,objects"
AZURE_TRANSLATOR_API_VERSION="3.0"
```

Use separate service-specific keys only if those services live in separate
Azure resources:

```bash
# AZURE_VISION_ENDPOINT="https://YOUR-SEPARATE-VISION-RESOURCE.cognitiveservices.azure.com/"
# AZURE_VISION_KEY="YOUR-SEPARATE-VISION-KEY"
# AZURE_TRANSLATOR_KEY="YOUR-SEPARATE-TRANSLATOR-KEY"
```

Optional Azure AI Search:

```bash
AZURE_AI_SEARCH_ENDPOINT="https://YOUR-SEARCH-SERVICE.search.windows.net"
AZURE_AI_SEARCH_KEY="YOUR-SEARCH-QUERY-KEY"
AZURE_AI_SEARCH_INDEX_NAME="YOUR-SEARCH-INDEX"
AZURE_AI_SEARCH_API_VERSION="2024-07-01"
AZURE_AI_SEARCH_QUERY_TYPE="simple"
```

Optional GPU VM CLIP worker:

```bash
CLIP_WORKER_URL="http://YOUR-VM-PUBLIC-IP:8000"
CLIP_WORKER_API_KEY="YOUR-CLIP-WORKER-SECRET"
CLIP_WORKER_MODEL="clip-ViT-B-32"
CLIP_WORKER_TIMEOUT_SECONDS="120"
```

Runtime paths:

```bash
ADS_LOCAL_DATA_DIR="data/raw"
ADS_BLOB_CACHE_DIR="indexes/blob_cache"
ADS_CLIP_MODEL_CACHE_DIR="indexes/model_cache/huggingface"
# ADS_RUNTIME_DIR="/home/agentic-document-search"
```

Do not commit `.env`.

## Ingest Documents

Local folder:

```bash
.venv/bin/python src/ingest.py \
  --data-dir data/raw \
  --output indexes/files_index.json
```

Azure Blob:

```bash
.venv/bin/python src/ingest.py \
  --source azure-blob \
  --blob-container-url "$AZURE_BLOB_CONTAINER_URL" \
  --output indexes/files_index.json
```

On Azure App Service, if the index is missing and `AZURE_BLOB_CONTAINER_URL` is
configured, the Streamlit app can create the Blob metadata index automatically.

## Run The App

Streamlit UI:

```bash
.venv/bin/streamlit run src/ui.py
```

CLI agent:

```bash
.venv/bin/python src/agent.py \
  --query "Find the telecom PDF with graphs" \
  --mode llm \
  --content-mode auto \
  --translator-mode auto \
  --top-k 3
```

Visual run:

```bash
.venv/bin/python src/agent.py \
  --query "I remember a slide with a blue graph" \
  --mode llm \
  --content-mode auto \
  --visual-prefilter clip \
  --visual-mode azure \
  --max-visual-files 3 \
  --max-visual-pages-per-file 5 \
  --max-clip-pages 10
```

Japanese query:

```bash
.venv/bin/python src/agent.py \
  --query "青いグラフがある資料を探して" \
  --mode llm \
  --translator-mode auto \
  --visual-mode azure
```

## Azure App Service Deployment

Use Azure App Service as the application execution platform.

Startup command:

```bash
python -m streamlit run src/ui.py --server.port 8000 --server.address 0.0.0.0
```

Set the values from `.env.example` as App Service environment variables. Do not
upload the local `.env` file.

Recommended App Service shape for the demo:

```text
Linux App Service
Python 3.12
Basic B1 or larger
Always On enabled when available
```

The base deployment uses `requirements.txt`. Keep `requirements-clip.txt`
outside the App Service deployment unless the plan can handle the heavier ML
install. For the cleaner architecture, run CLIP on the Azure VM worker instead.

## Azure VM CLIP Worker

The optional worker lives in `vm_worker/`.

It lets the architecture stay clean:

```text
App Service
  handles UI, orchestration, Azure OpenAI, Translator, Vision

Azure VM / GPU VM
  handles heavier local visual page ranking
```

Run manually on the VM:

```bash
cd vm_worker
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000
```

Set these on the worker:

```bash
AZURE_BLOB_CONTAINER_URL="https://YOUR-STORAGE.blob.core.windows.net/YOUR-CONTAINER?YOUR-SAS"
CLIP_WORKER_API_KEY="choose-a-random-secret"
CLIP_WORKER_DEVICE="auto"
CLIP_WORKER_CACHE_DIR="worker_cache"
```

Set these on App Service:

```bash
CLIP_WORKER_URL="http://YOUR-VM-PUBLIC-IP:8000"
CLIP_WORKER_API_KEY="same-random-secret"
```

Restrict VM inbound access:

```text
SSH
  your IP only

Port 8000
  App Service outbound IPs only
```

Deallocate the VM when the visual demo path is not being used.

## Current MVP Status

Done:

- File-level metadata ingestion
- Local and Azure Blob-ready index creation
- Metadata search with reasons
- Azure OpenAI metadata reranking
- Candidate-only text extraction for PDF, PPTX, DOCX, XLSX/XLSM
- Local OCR fallback for scanned PDF pages with thin text
- Deep Azure OpenAI content reranking
- Japanese/English mixed search support
- Japanese reason rewriting for Japanese prompts
- Search memory
- Lazy content, visual, and CLIP caches
- Local visual page skim
- Optional CLIP prefilter
- Optional Azure GPU VM CLIP worker
- Azure Vision verification for selected rendered PDF pages
- Optional Azure AI Search retrieval
- Semantic Kernel auto mode
- Streamlit UI for demo
- Azure App Service deployment path

Next architecture ideas:

- OneDrive / SharePoint connector through Microsoft Graph
- Native PPTX slide rendering for visual inspection
- Better page preview for the selected evidence page
- Azure Document Intelligence only for layout/table-heavy prompts
- More polished final answer generation after ranking

## Hackathon Narrative

Business impact:

```text
Employees waste time because they remember documents by partial meaning,
folder context, and visual memory, not exact filenames.
```

Approach:

```text
Agentic Document Search behaves like a human search partner:
it guesses likely files first, opens only candidates, checks visual evidence
only when needed, and explains why each result matched.
```

Operational value:

```text
The architecture is cost-aware and policy-aware.
Strict teams can use metadata/text-only modes.
Azure-approved teams can enable Foundry, Translator, Vision, and VM workers.
Documents can stay in approved cloud storage such as Blob, OneDrive, or
SharePoint instead of being copied into an uncontrolled AI system.
```
