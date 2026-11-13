import pytest
from fastapi.testclient import TestClient

from meshllm.server import (
    app, jobs, pair_sessions, workers, _worker_is_eligible, _worker_score,
)


@pytest.fixture(autouse=True)
def reset_state():
    workers.clear()
    jobs.clear()
    pair_sessions.clear()
    yield
    workers.clear()
    jobs.clear()
    pair_sessions.clear()


def register(client, **overrides):
    body = {
        "name": "Test device", "platform": "Test OS", "cpu_threads": 4,
        "ram_mb": 4096, "gpu_name": "Test GPU", "runtimes": [],
        "ollama_url": "http://127.0.0.1:11434", "rpc_endpoint": None,
        "max_parallel_jobs": 2,
    }
    body.update(overrides)
    session = client.post("/api/pairing-sessions", json=body)
    assert session.status_code == 200
    credentials = session.json()
    approved = client.post(
        f"/api/pairing-sessions/{credentials['session_id']}/approve",
        params={"worker_secret": credentials["worker_secret"]},
    )
    assert approved.status_code == 200
    return client.post("/api/workers/register", json={
        **body, "pairing_session_id": credentials["session_id"],
        "worker_secret": credentials["worker_secret"],
    })


def test_health():
    with TestClient(app) as client:
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


def test_qr_pairing_requires_owner_approval_and_is_single_use():
    with TestClient(app) as client:
        metadata = {
            "name": "Test device", "platform": "Test OS", "cpu_threads": 4,
            "ram_mb": 4096, "max_parallel_jobs": 1,
        }
        session = client.post("/api/pairing-sessions", json=metadata).json()
        payload = {**metadata, "pairing_session_id": session["session_id"],
                   "worker_secret": session["worker_secret"]}
        denied = client.post("/api/workers/register", json=payload)
        assert denied.status_code == 403
        approved = client.post(
            f"/api/pairing-sessions/{session['session_id']}/approve",
            params={"worker_secret": session["worker_secret"]},
        )
        assert approved.status_code == 200
        assert client.post("/api/workers/register", json=payload).status_code == 200
        assert client.post("/api/workers/register", json=payload).status_code == 404
        assert client.post("/api/pairing-token").status_code == 404


def test_qr_approval_page_shows_device_and_requires_secret():
    with TestClient(app) as client:
        metadata = {"name": "My laptop", "platform": "Linux", "cpu_threads": 8, "ram_mb": 8192}
        session = client.post("/api/pairing-sessions", json=metadata).json()
        path = f"/pair/approve/{session['session_id']}"
        assert client.get(path).status_code == 422
        page = client.get(path, params={"worker_secret": session["worker_secret"]})
        assert page.status_code == 200
        assert "My laptop" in page.text
        assert "Approve this device" in page.text


def test_health_exposes_build_and_server_timestamps():
    with TestClient(app) as client:
        health = client.get("/api/health").json()
        assert health["version"] == "0.3.0-dev"
        assert health["build_timestamp_utc"]
        assert health["server_time_utc"]


def test_worker_registers_rpc_endpoint_and_concurrency():
    with TestClient(app) as client:
        response = register(client, rpc_endpoint="192.168.1.22:50052", max_parallel_jobs=3)
        assert response.status_code == 200
        worker = client.get("/api/workers").json()[0]
        assert worker["rpc_endpoint"] == "192.168.1.22:50052"
        assert worker["max_parallel_jobs"] == 3
        assert worker["active_jobs"] == 0


def test_distributed_plan_uses_only_registered_rpc_workers():
    with TestClient(app) as client:
        assert register(client, rpc_endpoint="192.168.1.22:50052").status_code == 200
        plan = client.post("/api/distributed/plan", json={"model_path": "/models/test.gguf"})
        assert plan.status_code == 200
        assert "--rpc" in plan.json()["argv"]
        assert "192.168.1.22:50052" in plan.json()["command"]


def test_distributed_plan_requires_rpc_worker():
    with TestClient(app) as client:
        response = client.post("/api/distributed/plan", json={"model_path": "/models/test.gguf"})
        assert response.status_code == 409


def test_tensor_split_requires_one_positive_value_per_endpoint_plus_local():
    with TestClient(app) as client:
        register(client, rpc_endpoint="192.168.1.22:50052")
        bad = client.post("/api/distributed/plan", json={
            "model_path": "model.gguf", "tensor_split": "1,2,3"
        })
        assert bad.status_code == 422
        good = client.post("/api/distributed/plan", json={
            "model_path": "model.gguf", "tensor_split": "1,2"
        })
        assert good.status_code == 200


def test_tensor_split_rejects_non_finite_values():
    with TestClient(app) as client:
        register(client, rpc_endpoint="192.168.1.22:50052")
        response = client.post("/api/distributed/plan", json={
            "model_path": "model.gguf", "tensor_split": "1,nan"
        })
        assert response.status_code == 422



def test_registration_stores_initial_telemetry():
    with TestClient(app) as client:
        response = register(
            client, available_ram_mb=2500, cpu_percent=23.5,
            gpu_mem_total_mb=8192, gpu_mem_free_mb=6000,
            gpu_util_percent=18, battery_percent=88,
            battery_plugged=True, temperature_c=48.2,
        )
        assert response.status_code == 200
        worker = client.get("/api/workers").json()[0]
        assert worker["available_ram_mb"] == 2500
        assert worker["cpu_percent"] == 23.5
        assert worker["gpu_mem_free_mb"] == 6000
        assert worker["battery_plugged"] is True
        assert worker["temperature_c"] == 48.2
        assert worker["metrics_age_seconds"] is not None


def test_heartbeat_updates_live_telemetry():
    with TestClient(app) as client:
        response = register(client)
        worker_id = response.json()["worker_id"]
        heartbeat = client.post(
            f"/api/workers/{worker_id}/heartbeat",
            json={"available_ram_mb": 1536, "cpu_percent": 71.0, "temperature_c": 67.5},
        )
        assert heartbeat.status_code == 200
        worker = client.get("/api/workers").json()[0]
        assert worker["available_ram_mb"] == 1536
        assert worker["cpu_percent"] == 71.0
        assert worker["temperature_c"] == 67.5
        assert worker["metrics_updated_at"] is not None


def test_scheduler_ranks_device_with_more_headroom_higher():
    with TestClient(app) as client:
        assert register(
            client, name="Strong device", available_ram_mb=3000,
            cpu_percent=10, temperature_c=42, battery_percent=90,
            battery_plugged=False,
        ).status_code == 200
        assert register(
            client, name="Busy device", available_ram_mb=512,
            cpu_percent=90, temperature_c=70, battery_percent=15,
            battery_plugged=False,
        ).status_code == 200
        by_name = {worker.name: worker for worker in workers.values()}
        assert _worker_score(by_name["Strong device"]) > _worker_score(by_name["Busy device"])
        preview = client.get("/api/scheduler/preview").json()
        assert preview["eligible"][0]["name"] == "Strong device"


def test_scheduler_excludes_hot_and_critically_low_battery_devices():
    with TestClient(app) as client:
        assert register(
            client, name="Hot device", temperature_c=85.0,
        ).status_code == 200
        assert register(
            client, name="Low battery", battery_percent=5.0,
            battery_plugged=False,
        ).status_code == 200
        by_name = {worker.name: worker for worker in workers.values()}
        assert not _worker_is_eligible(by_name["Hot device"])
        assert not _worker_is_eligible(by_name["Low battery"])
        preview = client.get("/api/scheduler/preview").json()
        excluded = {item["name"]: item["reason"] for item in preview["excluded"]}
        assert "temperature safety limit" in excluded["Hot device"]
        assert "critical battery" in excluded["Low battery"]


def test_legacy_empty_heartbeat_remains_accepted():
    with TestClient(app) as client:
        worker_id = register(client).json()["worker_id"]
        response = client.post(f"/api/workers/{worker_id}/heartbeat")
        assert response.status_code == 200
