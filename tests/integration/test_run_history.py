import json

from fastapi.testclient import TestClient

from shar.api.app import create_app
from shar.errors import SharError
from shar.workers.runner import Worker


async def test_run_history_persists_details_and_results(settings, asset):
    app = create_app(settings)
    app.state.repo.add_video(asset)
    client = TestClient(app)
    body = {"video_id": asset.id, "prompt": "walking", "enable_refinement": False}
    first = client.post("/jobs", json=body).json()["job_id"]
    second = client.post("/jobs", json=body).json()["job_id"]
    assert first != second
    await Worker(settings).once()
    claimed = app.state.repo.claim()
    app.state.repo.fail(
        claimed["id"], claimed["lease_token"], SharError("TEST_FAILURE", "Test failure")
    )
    # A fresh app still sees both runs, including a failed run with no result.
    client = TestClient(create_app(settings))
    page = client.get("/jobs?limit=1").json()
    assert page["runs"][0]["id"] == second
    assert page["next_offset"] == 1
    assert client.get("/jobs?limit=1&offset=1").json()["runs"][0]["id"] == first
    failed = client.get(f"/jobs/{second}/details").json()
    assert failed["status"] == "failed"
    assert failed["error_message"] == "Test failure"
    assert failed["request"]["prompt"] == "walking"
    assert failed["activity"]
    assert not failed["has_result"]
    assert "lease_token" not in failed
    assert "storage_root" not in failed["config"]
    assert client.get(f"/jobs/{first}/details").json()["has_result"]
    assert client.get(f"/results/{first}").status_code == 200
    assert client.get("/jobs?limit=0").status_code == 422


def test_history_is_owner_scoped(settings, asset, monkeypatch):
    monkeypatch.setenv("SHAR_AUTH_TOKENS", json.dumps({"a": "local", "b": "other"}))
    app = create_app(settings.model_copy(update={"auth_mode": "token"}))
    app.state.repo.add_video(asset)
    client = TestClient(app)
    a, b = {"Authorization": "Bearer a"}, {"Authorization": "Bearer b"}
    job = client.post("/jobs", headers=a, json={"video_id": asset.id, "prompt": "walk"}).json()[
        "job_id"
    ]
    assert client.get("/jobs").status_code == 401
    assert len(client.get("/jobs", headers=a).json()["runs"]) == 1
    assert client.get("/jobs", headers=b).json()["runs"] == []
    assert client.get(f"/jobs/{job}/details", headers=b).status_code == 404
