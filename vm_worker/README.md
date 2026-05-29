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

## Azure Shape

Use this deployment shape for the hackathon demo:

```text
Azure App Service
  Streamlit UI + agent controller
  calls CLIP_WORKER_URL with X-Clip-Worker-Key

Azure GPU VM, Ubuntu
  FastAPI CLIP worker on port 8000
  downloads candidate PDFs from Azure Blob
  returns only selected visual page numbers

Azure Blob Storage
  shared document source

Azure Vision
  called by App Service after CLIP narrows the pages
```

Recommended VM direction:

```text
Image: Ubuntu LTS
Size: smallest available NVIDIA T4/A10 class GPU size that your quota supports
Inbound: SSH from your IP, worker port 8000 only from App Service outbound IPs
Auth: CLIP_WORKER_API_KEY
```

This keeps the expensive visual step separate from the web app while preserving
the human-like flow: the agent narrows candidate files, CLIP narrows pages, and
Azure Vision inspects only the most likely pages.

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

## Run As A Service

Manual `uvicorn` is fine while testing. For a stable demo, run the worker with
systemd:

```bash
cp .env.example .env
nano .env
sudo cp clip-worker.service.example /etc/systemd/system/clip-worker.service
sudo systemctl daemon-reload
sudo systemctl enable clip-worker
sudo systemctl start clip-worker
sudo systemctl status clip-worker
```

Edit `clip-worker.service.example` first if your repo path or Linux username is
not `/home/azureuser/agentic-document-search`.

## App Service Settings

Set these on the Streamlit App Service:

```bash
CLIP_WORKER_URL="http://YOUR-VM-PUBLIC-IP:8000"
CLIP_WORKER_API_KEY="same-random-secret"
CLIP_WORKER_MODEL="clip-ViT-B-32"
CLIP_WORKER_TIMEOUT_SECONDS="120"
```

## Network Security Group

1. Open the App Service in Azure Portal.
2. Go to `Networking`.
3. Copy `Outbound addresses`. If Azure shows possible outbound addresses too,
   keep those in mind because App Service outbound IPs can change after some
   platform or plan operations.
4. Open the VM network security group.
5. Add an inbound rule for TCP port `8000`.
6. Set `Source` to `IP Addresses`.
7. Paste the App Service outbound IP addresses as the source ranges.
8. Keep SSH restricted to your own IP address.

Do not leave port `8000` open to `Any` for a public demo. The API key is a
second guard, but the network rule should do the first filtering.

Deallocate the VM when you are not using the visual demo path, because GPU VMs
keep billing while they are running.
