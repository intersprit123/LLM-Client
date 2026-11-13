from fastapi.testclient import TestClient

from meshllm.server import app


def test_health():
    with TestClient(app) as client:
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


def test_pairing_token_is_single_use():
    with TestClient(app) as client:
        token = client.post("/api/pairing-token").json()["pairing_token"]
        body = {"name": "Test device", "platform": "Test OS", "cpu_threads": 4,
                "ram_mb": 4096, "gpu_name": None, "runtimes": [], "ollama_url": None,
                "pairing_token": token}
        assert client.post("/api/workers/register", json=body).status_code == 200
        assert client.post("/api/workers/register", json=body).status_code == 401


def test_job_requires_available_worker():
    with TestClient(app) as client:
        response = client.post("/api/jobs", json={"model": "demo", "prompt": "hello"})
        assert response.status_code == 200
        job_id = response.json()["job_id"]
        result = client.get(f"/api/jobs/{job_id}")
        assert result.status_code == 200
        assert result.json()["status"] in {"queued", "failed"}
