from fastapi.testclient import TestClient

from meshllm.server import app


def register(client, **overrides):
    token = client.post("/api/pairing-token").json()["pairing_token"]
    body = {
        "name": "Test device", "platform": "Test OS", "cpu_threads": 4,
        "ram_mb": 4096, "gpu_name": "Test GPU", "runtimes": [],
        "ollama_url": "http://127.0.0.1:11434", "rpc_endpoint": None,
        "max_parallel_jobs": 2, "pairing_token": token,
    }
    body.update(overrides)
    return client.post("/api/workers/register", json=body)


def test_health():
    with TestClient(app) as client:
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


def test_pairing_token_is_single_use():
    with TestClient(app) as client:
        token = client.post("/api/pairing-token").json()["pairing_token"]
        body = {"name": "Test device", "platform": "Test OS", "cpu_threads": 4,
                "ram_mb": 4096, "pairing_token": token}
        assert client.post("/api/workers/register", json=body).status_code == 200
        assert client.post("/api/workers/register", json=body).status_code == 401


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
