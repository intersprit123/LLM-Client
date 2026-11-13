"""MeshLLM coordinator: opt-in worker pool and llama.cpp RPC planner."""
from __future__ import annotations

import asyncio
import hmac
import math
import os
import secrets
import shlex
import time
import uuid
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

PAIR_TTL_SECONDS = 300
HEARTBEAT_TIMEOUT_SECONDS = 30
APP_VERSION = "0.3.0-dev"
APP_BUILD_TIMESTAMP = os.getenv(
    "MESHELL_BUILD_TIMESTAMP", datetime.now(timezone.utc).isoformat()
)
pair_sessions: dict[str, dict[str, Any]] = {}
MAX_PROMPT_CHARS = 16_000
MAX_OUTPUT_TOKENS = 2048


@dataclass
class Worker:
    worker_id: str
    name: str
    platform: str
    cpu_threads: int
    ram_mb: int
    gpu_name: str | None
    runtimes: list[str]
    ollama_url: str | None
    rpc_endpoint: str | None
    max_parallel_jobs: int
    active_jobs: int
    available_ram_mb: int | None
    cpu_percent: float | None
    gpu_mem_total_mb: int | None
    gpu_mem_free_mb: int | None
    gpu_util_percent: float | None
    battery_percent: float | None
    battery_plugged: bool | None
    temperature_c: float | None
    metrics_updated_at: float | None
    joined_at: float
    last_seen: float
    enabled: bool = True


@dataclass
class Job:
    job_id: str
    model: str
    prompt: str
    status: str
    created_at: float
    worker_id: str | None = None
    response: str | None = None
    error: str | None = None
    scheduler_score: float | None = None
    scheduler_decision: str | None = None


class PairRequest(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    platform: str = Field(min_length=1, max_length=40)
    cpu_threads: int = Field(ge=1, le=512)
    ram_mb: int = Field(ge=128, le=2_000_000)
    gpu_name: str | None = Field(default=None, max_length=128)
    runtimes: list[str] = Field(default_factory=list, max_length=16)
    ollama_url: str | None = Field(default=None, max_length=256)
    rpc_endpoint: str | None = Field(default=None, pattern=r"^[A-Za-z0-9.-]+:[0-9]{1,5}$")
    max_parallel_jobs: int = Field(default=1, ge=1, le=16)
    available_ram_mb: int | None = Field(default=None, ge=0, le=2_000_000)
    cpu_percent: float | None = Field(default=None, ge=0, le=100)
    gpu_mem_total_mb: int | None = Field(default=None, ge=0, le=2_000_000)
    gpu_mem_free_mb: int | None = Field(default=None, ge=0, le=2_000_000)
    gpu_util_percent: float | None = Field(default=None, ge=0, le=100)
    battery_percent: float | None = Field(default=None, ge=0, le=100)
    battery_plugged: bool | None = None
    temperature_c: float | None = Field(default=None, ge=-20, le=150)
    pairing_session_id: str = Field(min_length=16, max_length=128)
    worker_secret: str = Field(min_length=32, max_length=128)


class PairSessionRequest(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    platform: str = Field(min_length=1, max_length=40)
    cpu_threads: int = Field(ge=1, le=512)
    ram_mb: int = Field(ge=128, le=2_000_000)
    gpu_name: str | None = Field(default=None, max_length=128)
    runtimes: list[str] = Field(default_factory=list, max_length=16)
    ollama_url: str | None = Field(default=None, max_length=256)
    rpc_endpoint: str | None = Field(default=None, pattern=r"^[A-Za-z0-9.-]+:[0-9]{1,5}$")
    max_parallel_jobs: int = Field(default=1, ge=1, le=16)
    available_ram_mb: int | None = Field(default=None, ge=0, le=2_000_000)
    cpu_percent: float | None = Field(default=None, ge=0, le=100)
    gpu_mem_total_mb: int | None = Field(default=None, ge=0, le=2_000_000)
    gpu_mem_free_mb: int | None = Field(default=None, ge=0, le=2_000_000)
    gpu_util_percent: float | None = Field(default=None, ge=0, le=100)
    battery_percent: float | None = Field(default=None, ge=0, le=100)
    battery_plugged: bool | None = None
    temperature_c: float | None = Field(default=None, ge=-20, le=150)


class HeartbeatMetrics(BaseModel):
    available_ram_mb: int | None = Field(default=None, ge=0, le=2_000_000)
    cpu_percent: float | None = Field(default=None, ge=0, le=100)
    gpu_mem_total_mb: int | None = Field(default=None, ge=0, le=2_000_000)
    gpu_mem_free_mb: int | None = Field(default=None, ge=0, le=2_000_000)
    gpu_util_percent: float | None = Field(default=None, ge=0, le=100)
    battery_percent: float | None = Field(default=None, ge=0, le=100)
    battery_plugged: bool | None = None
    temperature_c: float | None = Field(default=None, ge=-20, le=150)


class JobRequest(BaseModel):
    model: str = Field(min_length=1, max_length=128)
    prompt: str = Field(min_length=1, max_length=MAX_PROMPT_CHARS)
    max_tokens: int = Field(default=512, ge=1, le=MAX_OUTPUT_TOKENS)


class DistributedPlanRequest(BaseModel):
    model_path: str = Field(min_length=1, max_length=512)
    host: str = Field(default="127.0.0.1", pattern=r"^[A-Za-z0-9.:-]+$")
    port: int = Field(default=8080, ge=1024, le=65535)
    parallel_slots: int = Field(default=2, ge=1, le=32)
    tensor_split: str | None = Field(default=None, max_length=256)


workers: dict[str, Worker] = {}
jobs: dict[str, Job] = {}
lock = asyncio.Lock()


async def _mark_stale() -> None:
    now = time.time()
    async with lock:
        for worker in workers.values():
            if now - worker.last_seen > HEARTBEAT_TIMEOUT_SECONDS:
                worker.enabled = False


def _worker_score(worker: Worker) -> float:
    """Higher is better: capacity and free slots, tempered by device health."""
    memory_headroom = (
        min(1.0, worker.available_ram_mb / max(worker.ram_mb, 1))
        if worker.available_ram_mb is not None else 0.5
    )
    cpu_headroom = (
        max(0.0, 1.0 - worker.cpu_percent / 100.0)
        if worker.cpu_percent is not None else 0.5
    )
    free_slots = max(
        0.0, 1.0 - worker.active_jobs / max(worker.max_parallel_jobs, 1)
    )
    battery_headroom = 1.0
    if worker.battery_percent is not None and not worker.battery_plugged:
        battery_headroom = 0.35 + 0.65 * worker.battery_percent / 100.0
    thermal_headroom = 1.0
    if worker.temperature_c is not None:
        thermal_headroom = max(0.25, min(1.0, (90.0 - worker.temperature_c) / 30.0))
    gpu_bonus = 0.0
    if worker.gpu_mem_total_mb and worker.gpu_mem_free_mb is not None:
        gpu_bonus = 0.03 * max(0.0, min(1.0, worker.gpu_mem_free_mb / worker.gpu_mem_total_mb))
    base = (
        0.36 * memory_headroom
        + 0.25 * cpu_headroom
        + 0.29 * free_slots
        + 0.10 * battery_headroom
    )
    return round(min(1.0, base * thermal_headroom + gpu_bonus), 4)


def _worker_exclusion_reason(worker: Worker) -> str | None:
    if not worker.enabled:
        return "offline or heartbeat timed out"
    if not worker.ollama_url:
        return "Ollama runtime not configured"
    if worker.active_jobs >= worker.max_parallel_jobs:
        return "at configured job capacity"
    if worker.temperature_c is not None and worker.temperature_c >= 85.0:
        return "temperature safety limit reached"
    if (worker.battery_percent is not None and worker.battery_percent <= 5.0
            and not worker.battery_plugged):
        return "critical battery while unplugged"
    return None


def _worker_is_eligible(worker: Worker) -> bool:
    return _worker_exclusion_reason(worker) is None


async def _run_job(job: Job, max_tokens: int) -> None:
    selected: Worker | None = None
    async with lock:
        candidates = [w for w in workers.values() if _worker_is_eligible(w)]
        candidates.sort(key=lambda w: (-_worker_score(w), w.name.lower()))
        if candidates:
            selected = candidates[0]
            job.scheduler_score = _worker_score(selected)
            selected.active_jobs += 1
            job.worker_id = selected.worker_id
            job.scheduler_decision = (
                "Selected highest health-aware capacity score; "
                "thermal and critical-battery limits were applied."
            )
            job.status = "running"

    if selected is None:
        job.status = "failed"
        job.error = "No available opted-in Ollama worker. Check worker status and concurrency limits."
        return

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=5.0)) as client:
            response = await client.post(
                selected.ollama_url.rstrip("/") + "/api/generate",
                json={"model": job.model, "prompt": job.prompt, "stream": False,
                      "options": {"num_predict": max_tokens}},
            )
            response.raise_for_status()
            job.response = str(response.json().get("response", ""))
            job.status = "completed"
    except (httpx.HTTPError, ValueError) as exc:
        job.status = "failed"
        job.error = f"Inference request failed: {type(exc).__name__}: {exc}"
    finally:
        async with lock:
            selected.active_jobs = max(0, selected.active_jobs - 1)


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield


app = FastAPI(title="MeshLLM Coordinator", version=APP_VERSION, lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def dashboard() -> str:
    return DASHBOARD


@app.get("/api/health")
async def health() -> dict[str, Any]:
    await _mark_stale()
    async with lock:
        return {
            "status": "ok", "version": APP_VERSION,
            "build_timestamp_utc": APP_BUILD_TIMESTAMP,
            "server_time_utc": datetime.now(timezone.utc).isoformat(),
            "workers": len(workers),
            "online_workers": sum(w.enabled for w in workers.values()),
            "jobs": len(jobs),
        }


@app.post("/api/pairing-sessions")
async def create_pairing_session(req: PairSessionRequest) -> dict[str, Any]:
    session_id = secrets.token_urlsafe(24)
    worker_secret = secrets.token_urlsafe(32)
    pair_sessions[session_id] = {
        "worker_secret": worker_secret,
        "approved": False,
        "expires_at": time.time() + PAIR_TTL_SECONDS,
        "metadata": req.model_dump(),
    }
    return {
        "session_id": session_id,
        "worker_secret": worker_secret,
        "expires_in_seconds": PAIR_TTL_SECONDS,
        "approval_path": f"/pair/approve/{session_id}",
    }


def _pair_session(session_id: str, worker_secret: str) -> dict[str, Any]:
    session = pair_sessions.get(session_id)
    if session is None or not hmac.compare_digest(session["worker_secret"], worker_secret):
        raise HTTPException(status_code=404, detail="Pairing request not found")
    if session["expires_at"] < time.time():
        pair_sessions.pop(session_id, None)
        raise HTTPException(status_code=410, detail="Pairing request expired")
    return session


@app.get("/api/pairing-sessions/{session_id}/status")
async def pairing_status(session_id: str, worker_secret: str) -> dict[str, Any]:
    session = _pair_session(session_id, worker_secret)
    return {"status": "approved" if session["approved"] else "waiting"}


@app.post("/api/pairing-sessions/{session_id}/approve")
async def approve_pairing(session_id: str, worker_secret: str) -> dict[str, bool]:
    session = _pair_session(session_id, worker_secret)
    session["approved"] = True
    session["approved_at"] = time.time()
    return {"approved": True}


@app.get("/pair/approve/{session_id}", response_class=HTMLResponse)
async def pairing_approval_page(session_id: str, worker_secret: str) -> str:
    session = _pair_session(session_id, worker_secret)
    metadata = session["metadata"]
    safe_name = (metadata["name"].replace("&", "&amp;").replace("<", "&lt;")
                 .replace(">", "&gt;").replace('"', "&quot;"))
    return f"""<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>Approve MeshLLM device</title><style>body{{font:16px system-ui;background:#0b1020;color:#eef;padding:28px;max-width:520px;margin:auto}}
    main{{background:#151d33;padding:24px;border-radius:16px}}button{{padding:14px 20px;background:#6d83ff;color:white;border:0;border-radius:10px;font-size:16px}}</style></head>
    <body><main><h1>Approve device?</h1><p><b>{safe_name}</b> wants to join your MeshLLM cluster.</p>
    <p>Platform: {metadata["platform"]}</p><p>CPU threads: {metadata["cpu_threads"]} · RAM: {metadata["ram_mb"]} MB</p>
    <p>This QR approval expires in five minutes. Only approve a device you control.</p>
    <button id="approve" onclick="approve()">Approve this device</button><p id="result"></p></main>
    <script>async function approve(){{let b=document.getElementById('approve');b.disabled=true;
    let r=await fetch('/api/pairing-sessions/{session_id}/approve?worker_secret={worker_secret}',{{method:'POST'}});
    document.getElementById('result').textContent=r.ok?'Approved. The worker can now join.':'Approval failed or expired.';}}</script></body></html>"""


@app.post("/api/workers/register")
async def register_worker(req: PairRequest) -> dict[str, Any]:
    session = _pair_session(req.pairing_session_id, req.worker_secret)
    if not session["approved"]:
        raise HTTPException(status_code=403, detail="Device has not been approved by QR")
    if session["metadata"] != req.model_dump(exclude={"pairing_session_id", "worker_secret"}):
        raise HTTPException(status_code=400, detail="Worker details changed after QR approval")
    pair_sessions.pop(req.pairing_session_id, None)
    if req.rpc_endpoint and not (1 <= int(req.rpc_endpoint.rsplit(":", 1)[1]) <= 65535):
        raise HTTPException(status_code=422, detail="RPC port must be between 1 and 65535")
    worker_id = str(uuid.uuid4())
    now = time.time()
    worker = Worker(
        worker_id=worker_id, name=req.name, platform=req.platform,
        cpu_threads=req.cpu_threads, ram_mb=req.ram_mb, gpu_name=req.gpu_name,
        runtimes=req.runtimes, ollama_url=req.ollama_url, rpc_endpoint=req.rpc_endpoint,
        max_parallel_jobs=req.max_parallel_jobs, active_jobs=0,
        available_ram_mb=req.available_ram_mb, cpu_percent=req.cpu_percent,
        gpu_mem_total_mb=req.gpu_mem_total_mb, gpu_mem_free_mb=req.gpu_mem_free_mb,
        gpu_util_percent=req.gpu_util_percent, battery_percent=req.battery_percent,
        battery_plugged=req.battery_plugged, temperature_c=req.temperature_c,
        metrics_updated_at=now if any(value is not None for value in (
            req.available_ram_mb, req.cpu_percent, req.gpu_mem_total_mb,
            req.gpu_mem_free_mb, req.gpu_util_percent, req.battery_percent,
            req.battery_plugged, req.temperature_c
        )) else None,
        joined_at=now, last_seen=now,
    )
    async with lock:
        workers[worker_id] = worker
    return {"worker_id": worker_id, "heartbeat_interval_seconds": 10}


@app.post("/api/workers/{worker_id}/heartbeat")
async def heartbeat(worker_id: str, metrics: HeartbeatMetrics | None = None) -> dict[str, bool]:
    async with lock:
        worker = workers.get(worker_id)
        if worker is None:
            raise HTTPException(status_code=404, detail="Unknown worker")
        worker.last_seen = time.time()
        worker.enabled = True
        if metrics is not None:
            for field, value in metrics.model_dump(exclude_unset=True).items():
                setattr(worker, field, value)
            if metrics.model_dump(exclude_unset=True):
                worker.metrics_updated_at = worker.last_seen
    return {"ok": True}


@app.post("/api/workers/{worker_id}/leave")
async def leave_worker(worker_id: str) -> dict[str, bool]:
    async with lock:
        worker = workers.pop(worker_id, None)
    if worker is None:
        raise HTTPException(status_code=404, detail="Unknown worker")
    return {"ok": True}


@app.get("/api/workers")
async def list_workers() -> list[dict[str, Any]]:
    await _mark_stale()
    async with lock:
        return [{
            **asdict(w),
            "online": w.enabled,
            "eligible_for_inference": _worker_is_eligible(w),
            "scheduler_score": _worker_score(w),
            "last_seen_ago_seconds": round(time.time() - w.last_seen, 1),
            "metrics_age_seconds": (
                round(time.time() - w.metrics_updated_at, 1)
                if w.metrics_updated_at is not None else None
            ),
        } for w in workers.values()]


@app.get("/api/scheduler/preview")
async def scheduler_preview() -> dict[str, Any]:
    """Explain which Ollama workers are eligible and how the policy ranks them."""
    await _mark_stale()
    async with lock:
        ranked = sorted(
            workers.values(),
            key=lambda w: (-_worker_score(w), w.name.lower()),
        )
        return {
            "policy": "automatic capacity scoring with thermal and critical-battery guards",
            "eligible": [
                {"worker_id": w.worker_id, "name": w.name,
                 "score": _worker_score(w), "active_jobs": w.active_jobs,
                 "max_parallel_jobs": w.max_parallel_jobs}
                for w in ranked if _worker_is_eligible(w)
            ],
            "excluded": [
                {"worker_id": w.worker_id, "name": w.name,
                 "reason": _worker_exclusion_reason(w)}
                for w in ranked if not _worker_is_eligible(w)
            ],
        }


@app.post("/api/jobs")
async def submit_job(req: JobRequest) -> dict[str, str]:
    job = Job(str(uuid.uuid4()), req.model, req.prompt, "queued", time.time())
    async with lock:
        jobs[job.job_id] = job
    asyncio.create_task(_run_job(job, req.max_tokens))
    return {"job_id": job.job_id, "status": job.status}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str) -> dict[str, Any]:
    async with lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job")
        return asdict(job)


@app.post("/api/distributed/plan")
async def distributed_plan(req: DistributedPlanRequest) -> dict[str, Any]:
    """Build a llama.cpp launch plan; never starts a process or opens RPC ports."""
    await _mark_stale()
    async with lock:
        endpoints = sorted({w.rpc_endpoint for w in workers.values()
                            if w.enabled and w.rpc_endpoint})
    if not endpoints:
        raise HTTPException(status_code=409, detail="No online worker has advertised a llama.cpp RPC endpoint")
    args = ["llama-server", "-m", req.model_path, "--host", req.host,
            "--port", str(req.port), "-np", str(req.parallel_slots),
            "--rpc", ",".join(endpoints)]
    if req.tensor_split:
        # Numeric comma-separated proportions only; avoid shell injection.
        parts = req.tensor_split.split(",")
        try:
            values = [float(part) for part in parts]
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="tensor_split must be comma-separated numbers") from exc
        if len(values) != len(endpoints) + 1 or any(not math.isfinite(v) or v <= 0 for v in values):
            raise HTTPException(status_code=422, detail="tensor_split needs one positive value per RPC worker plus the local host")
        args.extend(["--tensor-split", req.tensor_split])
    return {
        "backend": "llama.cpp RPC",
        "rpc_endpoints": endpoints,
        "command": shlex.join(args),
        "argv": args,
        "parallel_slots": req.parallel_slots,
        "note": "Review the command before running. llama.cpp RPC is proof-of-concept and insecure; use only on a trusted isolated LAN. The model file must be accessible on the coordinator.",
    }


DASHBOARD = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MeshLLM</title><style>
:root{color-scheme:dark;font-family:Inter,system-ui,sans-serif;background:#0b1020;color:#e8edff}
body{max-width:1050px;margin:40px auto;padding:0 20px}h1{font-size:2.2rem;margin-bottom:4px}
p{color:#9ca9c9}.card{background:#151d33;border:1px solid #293653;border-radius:16px;padding:20px;margin:16px 0}
.row{display:flex;gap:12px;flex-wrap:wrap}.stat{min-width:130px;flex:1}.big{font-size:2rem;font-weight:700}
button{background:#6d83ff;color:white;border:0;padding:11px 16px;border-radius:9px;cursor:pointer}
input,textarea{background:#0e1528;color:#e8edff;border:1px solid #344365;border-radius:8px;padding:10px;width:100%;box-sizing:border-box;margin:6px 0 12px}
textarea{min-height:110px}.muted{font-size:.9rem;color:#9ca9c9}table{width:100%;border-collapse:collapse}
td,th{text-align:left;padding:9px;border-bottom:1px solid #293653}
</style></head><body>
<h1>MeshLLM <span class="muted">0.3-dev</span></h1><p>Opt-in heterogeneous AI compute pool · trusted local network</p>
<div class="card"><b>Build / instance timestamp:</b> <span id="buildtime">Loading…</span><br><b>Current server time:</b> <span id="servertime">Loading…</span></div>
<div class="card row"><div class="stat"><div class="big" id="online">0</div><div class="muted">Online workers</div></div>
<div class="stat"><div class="big" id="total">0</div><div class="muted">Registered workers</div></div><div class="stat"><div class="big" id="jobs">0</div><div class="muted">Jobs tracked</div></div></div>
<div class="card"><h2>QR-only device approval</h2><p>Start the worker on the device you own. It displays a one-time QR code. Scan it with the cluster owner's phone and explicitly approve the device. Requests expire after five minutes.</p><p class="muted">No typed pairing tokens. Scanning alone does not grant access.</p></div>
<div class="card"><h2>Workers · live telemetry</h2><p class="muted">The scheduler continuously ranks eligible workers. Scores are relative capacity indicators, not benchmarked tokens/second.</p><table><thead><tr><th>Name</th><th>Platform</th><th>Threads</th><th>Free / total RAM</th><th>CPU load</th><th>GPU</th><th>Temp / battery</th><th>Jobs</th><th>Auto score</th></tr></thead><tbody id="workers"></tbody></table></div>
<div class="card"><h2>Submit parallel inference</h2><label>Ollama model</label><input id="model" value="llama3.2"><label>Prompt</label><textarea id="prompt" placeholder="Ask the model something..."></textarea><button onclick="submitJob()">Run inference</button><pre id="result"></pre></div>
<div class="card"><h2>Model-sharding launch plan</h2><p>Generate a llama.cpp RPC command using all paired workers that advertise an RPC endpoint. This does not launch it.</p>
<label>GGUF model path on coordinator</label><input id="gguf" placeholder="/models/model.gguf"><label>Parallel request slots</label><input id="slots" type="number" min="1" max="32" value="2"><button onclick="plan()">Generate distributed command</button><pre id="plan"></pre></div>
<script>
function fmtDate(value){return value?new Date(value).toLocaleString(undefined,{dateStyle:'medium',timeStyle:'medium',timeZoneName:'short'}):'Unavailable';}
async function refresh(){try{let [h,w]=await Promise.all([fetch('/api/health').then(r=>r.json()),fetch('/api/workers').then(r=>r.json())]);
document.getElementById('buildtime').textContent=fmtDate(h.build_timestamp_utc)+' (timestamp stored in UTC; displayed locally)';
document.getElementById('servertime').textContent=fmtDate(h.server_time_utc);
document.getElementById('online').textContent=h.online_workers;document.getElementById('total').textContent=h.workers;document.getElementById('jobs').textContent=h.jobs;
document.getElementById('workers').innerHTML=w.map(x=>'<tr><td>'+esc(x.name)+'</td><td>'+esc(x.platform)+'</td><td>'+x.cpu_threads+'</td><td>'+(x.available_ram_mb==null?'—':Math.round(x.available_ram_mb/1024)+' GB')+' / '+Math.round(x.ram_mb/1024)+' GB'</td><td>'+(x.cpu_percent==null?'—':Math.round(x.cpu_percent)+'%')+'</td><td>'+esc(x.gpu_name||'—')+(x.gpu_mem_free_mb==null?'':' ('+x.gpu_mem_free_mb+' MB free)')+'</td><td>'+(x.temperature_c==null?'—':x.temperature_c.toFixed(1)+'°C')+' / '+(x.battery_percent==null?'—':Math.round(x.battery_percent)+'%')+'</td><td>'+x.active_jobs+'/'+x.max_parallel_jobs+'</td><td>'+x.scheduler_score+' · '+(x.eligible_for_inference?'eligible':'held')+'</td></tr>').join('')||'<tr><td colspan="9">No workers paired yet.</td></tr>';
}catch(e){}}
function esc(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
async function submitJob(){let out=document.getElementById('result');out.textContent='Submitting...';let r=await fetch('/api/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:document.getElementById('model').value,prompt:document.getElementById('prompt').value})});let x=await r.json();if(!r.ok){out.textContent=JSON.stringify(x);return;}out.textContent='Job '+x.job_id+' queued...';let timer=setInterval(async()=>{let j=await fetch('/api/jobs/'+x.job_id).then(r=>r.json());if(j.status==='completed'||j.status==='failed'){clearInterval(timer);out.textContent=j.status==='completed'?j.response:(j.error||'Failed');}},1000);}
async function plan(){let out=document.getElementById('plan');out.textContent='Building plan...';let r=await fetch('/api/distributed/plan',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model_path:document.getElementById('gguf').value,parallel_slots:Number(document.getElementById('slots').value)})});let x=await r.json();out.textContent=r.ok?x.command+'\n\n'+x.note:JSON.stringify(x);}
refresh();setInterval(refresh,4000);
</script></body></html>"""
