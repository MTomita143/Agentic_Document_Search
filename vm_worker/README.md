# CLIP GPU VM Worker

This worker runs the visual page prefilter on an Azure GPU VM. The Streamlit
App Service sends candidate Blob names, the worker downloads those PDFs from
the shared Blob container, renders a bounded page sample, ranks pages with
CLIP, and returns the page numbers Azure Vision should inspect.

## Environment

```bash
AZURE_BLOB_CONTAINER_URL="https://YOUR-STORAGE.blob.core.windows.net/YOUR-CONTAINER?YOUR-SAS"
CLIP_WORKER_API_KEY="choose-a-random-secret"
CLIP_WORKER_DEVICE="auto"
CLIP_WORKER_CACHE_DIR="worker_cache"
```

`AZURE_BLOB_CONTAINER_URL` should be the same container URL used by the App
Service. The SAS needs read access. List access is useful for ingestion, but
the worker only needs to download known blobs.

## Run On The VM

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000
```

For a real GPU run, install the CUDA-enabled PyTorch build that matches the VM
image before installing `requirements.txt`. If CUDA is visible to PyTorch,
`CLIP_WORKER_DEVICE=auto` uses `cuda`; otherwise it falls back to CPU.

## App Service Settings

Set these on the Streamlit App Service:

```bash
CLIP_WORKER_URL="http://YOUR-VM-PUBLIC-IP:8000"
CLIP_WORKER_API_KEY="same-random-secret"
CLIP_WORKER_MODEL="clip-ViT-B-32"
CLIP_WORKER_TIMEOUT_SECONDS="120"
```

Keep the VM network narrow: restrict inbound access to your App Service
outbound IP addresses when possible. Deallocate the VM when you are not using
the visual demo path, because GPU VMs keep billing while they are running.
