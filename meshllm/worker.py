"""Opt-in MeshLLM worker. It shares bounded inference only; no remote shell or file access."""
from __future__ import annotations

import argparse
import asyncio
import os
import platform
import subprocess
import sys

import httpx


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


async def main() -> None:
    parser = argparse.ArgumentParser(description="Join a MeshLLM pool with the owner's consent.")
    parser.add_argument("--server", required=True, help="Coordinator URL, e.g. http://192.168.1.10:8765")
    parser.add_argument("--name", default=platform.node())
    parser.add_argument("--ollama-url", default=None, help="Local Ollama URL, e.g. http://127.0.0.1:11434")
    parser.add_argument("--rpc-endpoint", default=None, help="Manually configured llama.cpp RPC endpoint, e.g. 192.168.1.22:50052")
    parser.add_argument("--max-parallel-jobs", type=int, default=1, choices=range(1, 17))
    args = parser.parse_args()

    print("MeshLLM worker — explicit opt-in")
    print(f"Device: {args.name} ({platform.system()} {platform.machine()})")
    print(f"CPU threads: {os.cpu_count() or 1}; RAM: {memory_mb()} MB")
    print(f"GPU detected: {gpu_name() or 'not detected'}")
    print(f"Ollama: {args.ollama_url or 'disabled'}; llama.cpp RPC: {args.rpc_endpoint or 'not configured'}")
    print("RPC exposes compute, not a secure authenticated tunnel. Use only on a trusted isolated LAN.")
    if input("Join this coordinator and allow the configured workloads? [y/N] ").strip().lower() not in {"y", "yes"}:
        print("Not joining. No resources will be shared.")
        return

    pairing_token = input("Paste one-time pairing token: ").strip()
    if len(pairing_token) < 16:
        raise SystemExit("Pairing token looks invalid.")

    base = args.server.rstrip("/")
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(base + "/api/workers/register", json={
            "name": args.name, "platform": f"{platform.system()} {platform.machine()}",
            "cpu_threads": os.cpu_count() or 1, "ram_mb": memory_mb(),
            "gpu_name": gpu_name(),
            "runtimes": (["ollama"] if args.ollama_url else []) + (["llama.cpp-rpc"] if args.rpc_endpoint else []),
            "ollama_url": args.ollama_url, "rpc_endpoint": args.rpc_endpoint,
            "max_parallel_jobs": args.max_parallel_jobs, "pairing_token": pairing_token,
        })
        response.raise_for_status()
        worker_id = response.json()["worker_id"]
        print(f"Joined. Worker ID: {worker_id}. Press Ctrl+C to leave.")
        try:
            while True:
                await client.post(base + f"/api/workers/{worker_id}/heartbeat")
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
