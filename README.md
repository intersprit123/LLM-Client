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

Open `http://127.0.0.1:8765`, generate a one-time pairing token, and start each worker manually:

```bash
python -m meshllm.worker --server http://COORDINATOR_IP:8765 --name "Laptop" --ollama-url http://127.0.0.1:11434 --max-parallel-jobs 2
```

For a worker that also runs a separately configured llama.cpp RPC server:

```bash
python -m meshllm.worker --server http://COORDINATOR_IP:8765 --name "GPU-PC" --ollama-url http://127.0.0.1:11434 --rpc-endpoint 192.168.1.22:50052
```

On each remote RPC host, build llama.cpp with `-DGGML_RPC=ON` and the desired backend (CUDA, Metal, Vulkan, or CPU as supported by that build), then start `ggml-rpc-server` bound only to the trusted LAN interface. Check the upstream llama.cpp RPC documentation for the exact build and runtime options.

In the dashboard's model-sharding panel, enter a GGUF model path on the coordinator and generate a launch command. Review it before running. The generated command includes every online worker that registered an RPC endpoint:

```bash
llama-server -m /models/model.gguf --host 127.0.0.1 --port 8080 -np 2 --rpc 192.168.1.22:50052,192.168.1.23:50052
```

llama.cpp distributes model weights and KV cache across available devices by memory capacity by default; optional tensor split proportions can be supplied to the planning API.

## API

- `GET /api/health`
- `POST /api/pairing-token`
- `POST /api/workers/register`
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

The coordinator continuously ranks available Ollama workers using free-memory headroom, CPU headroom, free job slots and battery state. It does not ask the user to assign manual percentages. It excludes workers at or above 85°C and workers at 5% battery or lower when unplugged. These are conservative initial thresholds, not a substitute for device firmware thermal protection. The policy is a first milestone and will be tuned with real mixed-device benchmarks.

## Platform reality

The coordinator and Python worker can run on Windows, Linux and Android/Termux where Python and the dependencies are available. Actual GPU acceleration depends on the separately built inference runtime and drivers. A phone may contribute CPU/RAM to compatible backends, but it is not automatically a CUDA/Metal device. Heterogeneous pooling is best-effort and network bandwidth/latency can make a cluster slower than one machine.

## Development

Run tests with `pip install -r requirements-dev.txt && pytest`.
