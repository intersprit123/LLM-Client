# MeshLLM

MeshLLM is an opt-in, local-network compute pool for running AI workloads across computers and mobile devices.

> **Consent and safety:** Every worker must be explicitly paired, approved, and started by its owner. Workers advertise capabilities, accept bounded jobs, and can pause or disconnect at any time. No stealth installation, persistence, forced resource use, or remote shell is supported.

## MVP architecture

- **Coordinator (FastAPI):** owns the pool, worker registry, health/status reporting, and job lifecycle.
- **Worker agent:** explicitly connects to a coordinator, reports CPU/RAM/runtime capabilities, and exposes a bounded execution interface.
- **Web UI:** a lightweight dashboard for worker enrollment, pool status, and job submission.
- **Inference runtime:** model execution is initially delegated to a local Ollama endpoint. This keeps model compatibility and hardware acceleration inside a supported runtime while the project proves cross-device orchestration.

## Important limitation

A model cannot generally be split across arbitrary CPUs/GPUs and devices merely by adding their advertised FLOPS. Efficient tensor/model parallelism requires compatible runtimes, communication collectives, memory placement, and often matching accelerator backends. This MVP coordinates workers and routes independent inference requests; it does **not** yet combine separate GPUs into one unified VRAM device or shard a single model across heterogeneous phones and PCs.

## Quick start

Requirements: Python 3.11+

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
uvicorn meshllm.server:app --host 0.0.0.0 --port 8765
```

Open `http://127.0.0.1:8765`. Keep the coordinator on a trusted LAN. The API has a pairing token; do not expose it to the public internet.

Start a worker on another device:

```bash
python -m meshllm.worker --server http://COORDINATOR_LAN_IP:8765 --name "Bedroom-PC"
```

The worker prompts for a one-time pairing token, then asks for explicit local consent before joining. To execute inference, install Ollama on the worker and pull a model, then start the worker with `--ollama-url http://127.0.0.1:11434`.

## API

- `GET /` dashboard
- `GET /api/health`
- `POST /api/pairing-token` creates a short-lived token
- `POST /api/workers/register` enrolls an opted-in worker
- `POST /api/workers/{worker_id}/heartbeat`
- `POST /api/workers/{worker_id}/leave`
- `GET /api/workers`
- `POST /api/jobs` submits a prompt/model request
- `GET /api/jobs/{job_id}`

## Roadmap

1. LAN pairing, explicit worker consent, heartbeats, and dashboard.
2. Work-stealing request scheduler with model-aware routing and bounded concurrency.
3. Cross-platform worker packaging (Windows, Linux, Android/Termux).
4. Optional distributed inference backend (e.g. llama.cpp RPC / compatible tensor-parallel engine), validated per backend and hardware combination.
