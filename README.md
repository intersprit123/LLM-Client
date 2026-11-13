# MeshLLM

MeshLLM is an opt-in, local-network compute pool for running AI workloads across computers, tablets and phones.

> **Consent:** every worker is explicitly paired and started by its owner. Workers can leave at any time. MeshLLM does not provide stealth installation, remote shell, or forced resource use.

## Two compute modes

1. **Parallel requests (heterogeneous):** the coordinator balances independent inference requests across opted-in Ollama workers. This increases throughput, not the size of one model.
2. **Model sharding (llama.cpp RPC):** supported llama.cpp builds can expose CPU/GPU backends from remote devices. A coordinator-side `llama-server` uses `--rpc host:port,...` to distribute model weights and KV cache over those devices. The main host must have a compatible llama.cpp build and access to the model file.

llama.cpp RPC is explicitly documented upstream as proof-of-concept, fragile and insecure. Do not expose it to the internet or an untrusted LAN. Keep it on a trusted isolated network. The worker agent does not start or expose RPC automatically; the owner configures and starts the RPC server separately.

## Quick start

Python 3.11+:

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
uvicorn meshllm.server:app --host 0.0.0.0 --port 8765
```

Open `http://127.0.0.1:8765`. Start each worker manually; it creates a short-lived QR enrollment request and prints the QR in the terminal:

```bash
python -m meshllm.worker --server http://COORDINATOR_IP:8765 --name "Laptop" --ollama-url http://127.0.0.1:11434 --max-parallel-jobs 2
```

Scan the terminal QR with the cluster owner's phone and tap **Approve this device** on the confirmation page. The worker waits for approval and only registers after approval. QR requests expire after five minutes and are single-use. No pairing token is typed or displayed as a separate code. Use HTTPS where possible; on plain HTTP, the approval URL is a temporary bearer credential and must stay on a trusted LAN.

For a worker that also runs a separately configured llama.cpp RPC server:

```bash
python -m meshllm.worker --server http://COORDINATOR_IP:8765 --name "GPU-PC" --ollama-url http://127.0.0.1:11434 --rpc-endpoint 192.168.1.22:50052
```

On each remote RPC host, build llama.cpp with `-DGGML_RPC=ON` and the desired backend (CUDA, Metal, Vulkan, or CPU as supported by that build), then start `ggml-rpc-server` bound only to the trusted LAN interface. Check the upstream llama.cpp RPC documentation for the exact build and runtime options.

In the dashboard's model-sharding panel, enter a GGUF model path on the coordinator and generate a launch command. Review it before running. The generated command includes every online worker that registered an RPC endpoint:

```bash
llama-server -m /models/model.gguf --host 127.0.0.1 --port 8080 -np 2 --rpc 192.168.1.22:50052,192.168.1.23:50052
```

MeshLLM now generates an automatic tensor-split estimate when no manual split is supplied. It uses the coordinator's configurable `local_memory_mb` (default 4096 MB) plus each RPC worker's reported free VRAM, falling back to available RAM and then half of total RAM. The estimate is a planning aid, not a guarantee that a model will fit: backend device ordering, runtime reservations, KV cache and model architecture still matter. You can override it with explicit `tensor_split` proportions. The dashboard/API returns the memory assumptions alongside the command.

## API

- `GET /api/health`
- `POST /api/pairing-sessions`
- `GET /api/pairing-sessions/{session_id}/status`
- `POST /api/pairing-sessions/{session_id}/approve`
- `GET /pair/approve/{session_id}`
- `POST /api/workers/register` — requires a single-use, owner-approved QR session
- `POST /api/workers/{worker_id}/heartbeat`
- `POST /api/workers/{worker_id}/leave`
- `GET /api/workers` — includes latest telemetry, eligibility and scheduler score
- `GET /api/scheduler/preview` — explains the automatic ranking and exclusion decisions
- `POST /api/jobs` and `GET /api/jobs/{job_id}`
- `POST /api/distributed/plan` — creates a reviewed llama.cpp launch plan; it never launches a process.

## Automatic resource decisions

After the owner approves pairing, a worker sends CPU load, available RAM, supported GPU memory/utilization, battery state and temperature at registration and on each 10-second heartbeat. Sensor availability depends on the operating system and drivers; unsupported values stay unknown. Installing optional `psutil` can improve CPU, battery and temperature reporting on platforms where it is supported:

```bash
python -m pip install psutil
```

Workers also run a small CPU throughput probe at startup and measure coordinator round-trip time during heartbeats. These are coarse scheduling signals, not a substitute for benchmarking actual model tokens/second or GPU kernels. The coordinator ranks available Ollama workers using free-memory headroom, CPU headroom, free job slots, battery state, CPU probe score and network round-trip time. It does not ask the user to assign manual percentages. It excludes workers at or above 85°C and workers at 5% battery or lower when unplugged. These are conservative initial thresholds, not a substitute for device firmware thermal protection. The policy is a first milestone and will be tuned with real mixed-device benchmarks.

## Platform reality

The coordinator and Python worker can run on Windows, Linux and Android/Termux where Python and the dependencies are available. Actual GPU acceleration depends on the separately built inference runtime and drivers. A phone may contribute CPU/RAM to compatible backends, but it is not automatically a CUDA/Metal device. Heterogeneous pooling is best-effort and network bandwidth/latency can make a cluster slower than one machine.

## Development

The dashboard shows the build/instance timestamp in UTC, converted to the viewer's local timezone, plus the live server time. Set `MESHELL_BUILD_TIMESTAMP` to an ISO-8601 UTC timestamp in deployments to preserve the exact artifact build time; without it, the coordinator records its process initialization time.

Run tests with `pip install -r requirements-dev.txt && pytest`.
