"""Opt-in MeshLLM worker. It shares bounded inference only; no remote shell or file access."""
from __future__ import annotations

import argparse
import asyncio
import glob
import hashlib
import os
import platform
import subprocess
import sys
import time

import httpx
import qrcode


def memory_mb() -> int:
    try:
        if sys.platform == "win32":
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("memoryLoad", ctypes.c_ulong),
                            ("totalPhys", ctypes.c_ulonglong), ("availPhys", ctypes.c_ulonglong),
                            ("totalPageFile", ctypes.c_ulonglong), ("availPageFile", ctypes.c_ulonglong),
                            ("totalVirtual", ctypes.c_ulonglong), ("availVirtual", ctypes.c_ulonglong),
                            ("availExtendedVirtual", ctypes.c_ulonglong)]
            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return int(status.totalPhys / (1024 * 1024))
        if sys.platform == "darwin":
            return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()) // (1024 * 1024)
        with open("/proc/meminfo", encoding="utf-8") as f:
            pages = int(f.read().split("MemTotal:")[1].split()[0])
        return pages // 1024
    except Exception:
        return 0


def gpu_name() -> str | None:
    try:
        if sys.platform == "win32":
            cmd = ["powershell", "-NoProfile", "-Command",
                   "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"]
        else:
            cmd = ["sh", "-lc", "(command -v lspci >/dev/null && lspci | grep -Ei 'vga|3d|display') || true"]
        output = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=4)
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        return ", ".join(lines[:3])[:128] or None
    except Exception:
        return None


def available_memory_mb() -> int | None:
    try:
        if sys.platform == "win32":
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("memoryLoad", ctypes.c_ulong),
                            ("totalPhys", ctypes.c_ulonglong), ("availPhys", ctypes.c_ulonglong),
                            ("totalPageFile", ctypes.c_ulonglong), ("availPageFile", ctypes.c_ulonglong),
                            ("totalVirtual", ctypes.c_ulonglong), ("availVirtual", ctypes.c_ulonglong),
                            ("availExtendedVirtual", ctypes.c_ulonglong)]
            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.availPhys / (1024 * 1024))
            return None
        if sys.platform == "darwin":
            output = subprocess.check_output(["vm_stat"], text=True, timeout=3)
            page_size = int(output.split("page size of ")[1].split(" bytes")[0])
            pages = 0
            for line in output.splitlines():
                if any(key in line for key in ("Pages free:", "Pages inactive:", "Pages speculative:")):
                    pages += int(line.split(":")[1].strip().rstrip("."))
            return pages * page_size // (1024 * 1024)
        with open("/proc/meminfo", encoding="utf-8") as f:
            values = {}
            for line in f:
                key, value = line.split(":", 1)
                if key in {"MemAvailable", "MemFree"}:
                    values[key] = int(value.split()[0])
            return values.get("MemAvailable", values.get("MemFree", 0)) // 1024
    except Exception:
        return None


def cpu_percent() -> float | None:
    try:
        import psutil
        return round(float(psutil.cpu_percent(interval=0.15)), 1)
    except ImportError:
        pass
    except Exception:
        return None

    if hasattr(os, "getloadavg") and os.cpu_count():
        try:
            return round(min(100.0, max(0.0, os.getloadavg()[0] / os.cpu_count() * 100)), 1)
        except OSError:
            return None
    if sys.platform.startswith("linux"):
        try:
            def read_times():
                with open("/proc/stat", encoding="utf-8") as f:
                    fields = [int(x) for x in f.readline().split()[1:]]
                return sum(fields), fields[3] + (fields[4] if len(fields) > 4 else 0)
            total_a, idle_a = read_times()
            time.sleep(0.12)
            total_b, idle_b = read_times()
            total_delta, idle_delta = total_b - total_a, idle_b - idle_a
            if total_delta > 0:
                return round(100.0 * (1 - idle_delta / total_delta), 1)
        except Exception:
            return None
    return None


def gpu_metrics() -> dict:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL, timeout=4,
        )
        rows = [line.split(",") for line in output.splitlines() if line.strip()]
        if not rows:
            return {}
        first = [value.strip() for value in rows[0]]
        return {
            "gpu_name": first[0][:128],
            "gpu_mem_total_mb": max(0, int(float(first[1]))),
            "gpu_mem_free_mb": max(0, int(float(first[2]))),
            "gpu_util_percent": max(0.0, min(100.0, float(first[3]))),
        }
    except Exception:
        return {}


def battery_metrics() -> dict:
    try:
        import psutil
        battery = psutil.sensors_battery()
        if battery is not None:
            return {
                "battery_percent": round(float(battery.percent), 1),
                "battery_plugged": bool(battery.power_plugged),
            }
    except Exception:
        pass
    try:
        for base in glob.glob("/sys/class/power_supply/*"):
            if not os.path.exists(os.path.join(base, "capacity")):
                continue
            with open(os.path.join(base, "capacity"), encoding="utf-8") as f:
                percent = float(f.read().strip())
            status_path = os.path.join(base, "status")
            status = ""
            if os.path.exists(status_path):
                with open(status_path, encoding="utf-8") as f:
                    status = f.read().strip().lower()
            return {"battery_percent": percent, "battery_plugged": status in {"charging", "full"}}
    except Exception:
        pass
    return {}


def temperature_c() -> float | None:
    readings = []
    for path in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
        try:
            with open(path, encoding="utf-8") as f:
                raw = float(f.read().strip())
            value = raw / 1000.0 if raw > 200 else raw
            if -20 <= value <= 150:
                readings.append(value)
        except (OSError, ValueError):
            continue
    if readings:
        return round(max(readings), 1)
    try:
        import psutil
        sensors = psutil.sensors_temperatures()
        values = [entry.current for group in sensors.values() for entry in group
                  if entry.current is not None and -20 <= entry.current <= 150]
        return round(max(values), 1) if values else None
    except Exception:
        return None


def cpu_benchmark_score() -> float | None:
    """Small repeatable CPU throughput probe; not a substitute for model benchmarks."""
    rounds = 40_000
    started = time.perf_counter()
    try:
        hashlib.pbkdf2_hmac("sha256", b"MeshLLM CPU benchmark", b"meshllm-v1", rounds)
        elapsed = max(time.perf_counter() - started, 0.000001)
        return round(rounds / elapsed, 1)
    except Exception:
        return None


def collect_telemetry() -> dict:
    telemetry = {
        "available_ram_mb": available_memory_mb(),
        "cpu_percent": cpu_percent(),
        "temperature_c": temperature_c(),
        "cpu_benchmark_score": cpu_benchmark_score(),
    }
    telemetry.update(gpu_metrics())
    telemetry.update(battery_metrics())
    return {key: value for key, value in telemetry.items() if value is not None}


async def main() -> None:
    parser = argparse.ArgumentParser(description="Join a MeshLLM pool with the owner's consent.")
    parser.add_argument("--server", required=True, help="Coordinator URL, e.g. http://192.168.1.10:8765")
    parser.add_argument("--name", default=platform.node())
    parser.add_argument("--ollama-url", default=None, help="Local Ollama URL, e.g. http://127.0.0.1:11434")
    parser.add_argument("--rpc-endpoint", default=None, help="Manually configured llama.cpp RPC endpoint, e.g. 192.168.1.22:50052")
    parser.add_argument("--max-parallel-jobs", type=int, default=1, choices=range(1, 17))
    args = parser.parse_args()

    telemetry = collect_telemetry()
    detected_gpu = telemetry.get("gpu_name") or gpu_name()
    print("MeshLLM worker — explicit opt-in")
    print(f"Device: {args.name} ({platform.system()} {platform.machine()})")
    print(f"CPU threads: {os.cpu_count() or 1}; RAM: {memory_mb()} MB total")
    print(f"Available RAM: {telemetry.get('available_ram_mb', 'unknown')} MB; "
          f"CPU load: {telemetry.get('cpu_percent', 'unknown')}%")
    print(f"GPU detected: {detected_gpu or 'not detected'}")
    print(f"Temperature: {telemetry.get('temperature_c', 'unknown')}°C; "
          f"battery: {telemetry.get('battery_percent', 'unknown')}%")
    print(f"Ollama: {args.ollama_url or 'disabled'}; llama.cpp RPC: {args.rpc_endpoint or 'not configured'}")
    print("QR approval required. No worker joins until the cluster owner approves it.")
    base = args.server.rstrip("/")
    metadata = {
        "name": args.name, "platform": f"{platform.system()} {platform.machine()}",
        "cpu_threads": os.cpu_count() or 1, "ram_mb": memory_mb(),
        "gpu_name": detected_gpu, **telemetry,
        "runtimes": (["ollama"] if args.ollama_url else []) + (["llama.cpp-rpc"] if args.rpc_endpoint else []),
        "ollama_url": args.ollama_url, "rpc_endpoint": args.rpc_endpoint,
        "max_parallel_jobs": args.max_parallel_jobs,
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        session_response = await client.post(base + "/api/pairing-sessions", json=metadata)
        session_response.raise_for_status()
        session = session_response.json()
        approval_url = base + session["approval_path"] + "?worker_secret=" + session["worker_secret"]
        qr = qrcode.QRCode(border=2)
        qr.add_data(approval_url)
        qr.make(fit=True)
        print("\\nScan this QR with the cluster owner's phone, then tap Approve:")
        qr.print_ascii(out=sys.stdout)
        print(f"Waiting for approval (expires in {session['expires_in_seconds']} seconds)...")
        deadline = time.monotonic() + session["expires_in_seconds"]
        while time.monotonic() < deadline:
            status_response = await client.get(
                base + f"/api/pairing-sessions/{session['session_id']}/status",
                params={"worker_secret": session["worker_secret"]},
            )
            if status_response.status_code == 410:
                raise SystemExit("QR approval expired. Restart the worker to create a new QR.")
            status_response.raise_for_status()
            if status_response.json()["status"] == "approved":
                break
            await asyncio.sleep(2)
        else:
            raise SystemExit("QR approval timed out. Restart the worker to create a new QR.")

        response = await client.post(base + "/api/workers/register", json={
            **metadata,
            "pairing_session_id": session["session_id"],
            "worker_secret": session["worker_secret"],
        })
        response.raise_for_status()
        worker_id = response.json()["worker_id"]
        print(f"Joined. Worker ID: {worker_id}. Press Ctrl+C to leave.")
        try:
            while True:
                heartbeat_metrics = collect_telemetry()
                started = time.perf_counter()
                await client.post(
                    base + f"/api/workers/{worker_id}/heartbeat",
                    json=heartbeat_metrics,
                )
                heartbeat_metrics["network_rtt_ms"] = round(
                    (time.perf_counter() - started) * 1000, 2
                )
                # Include measured round-trip time in the next heartbeat.
                await client.post(
                    base + f"/api/workers/{worker_id}/heartbeat",
                    json={"network_rtt_ms": heartbeat_metrics["network_rtt_ms"]},
                )
                await asyncio.sleep(10)
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            try:
                await client.post(base + f"/api/workers/{worker_id}/leave")
            except httpx.HTTPError:
                pass
            print("Worker left the pool.")


if __name__ == "__main__":
    asyncio.run(main())
