import json

from fastapi.testclient import TestClient

from shar.api.app import create_app


def test_owner_isolation_and_urls(settings, asset, monkeypatch):
    monkeypatch.setenv("SHAR_AUTH_TOKENS", json.dumps({"token-a": "local", "token-b": "other"}))
    app = create_app(settings.model_copy(update={"auth_mode": "token"}))
    app.state.repo.add_video(asset)
    client = TestClient(app)
    assert client.get("/videos/" + asset.id + "/content").status_code == 401
    assert (
        client.get(
            "/videos/" + asset.id + "/content", headers={"Authorization": "Bearer token-b"}
        ).status_code
        == 404
    )
    response = client.post(
        "/jobs",
        json={"video_url": "http://127.0.0.1/private", "prompt": "x"},
        headers={"Authorization": "Bearer token-a"},
    )
    assert response.status_code == 422
    assert (
        client.get(
            "/videos/" + asset.id + "/content", headers={"Authorization": "Bearer token-a"}
        ).status_code
        == 200
    )


def test_upload_limit_and_config_override(settings, media):
    small = settings.model_copy(update={"max_upload_bytes": 100})
    client = TestClient(create_app(small))
    with (media / "cfr.mp4").open("rb") as f:
        response = client.post("/videos", files={"file": ("x.mp4", f, "video/mp4")})
    assert response.status_code == 413, response.text
    response = client.post(
        "/jobs", json={"video_id": "x", "prompt": "x", "storage_root": "/tmp/escape"}
    )
    assert response.status_code == 422


def test_stream_events_are_owner_scoped_and_incremental(settings, monkeypatch):
    monkeypatch.setenv("SHAR_AUTH_TOKENS", json.dumps({"token-a": "local", "token-b": "other"}))
    app = create_app(settings.model_copy(update={"auth_mode": "token"}))
    client = TestClient(app)
    session = app.state.repo.start_stream("local", "operator-camera")
    app.state.repo.append_stream_event(session, "local", {"type": "event", "revision": 1})
    path = f"/streams/{session}/events"
    assert client.get(path, headers={"Authorization": "Bearer token-b"}).status_code == 404
    result = client.get(path, headers={"Authorization": "Bearer token-a"}).json()["events"]
    assert result[0]["revision"] == 1
    assert (
        client.get(
            path + "?after=" + str(result[-1]["sequence"]),
            headers={"Authorization": "Bearer token-a"},
        ).json()["events"]
        == []
    )


def test_body_limit_precedes_multipart_spooling(settings):
    client = TestClient(create_app(settings.model_copy(update={"max_upload_bytes": 100})))
    response = client.post(
        "/videos",
        content=b"x" * 100000,
        headers={"Content-Type": "multipart/form-data; boundary=x"},
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "VIDEO_TOO_LARGE"


def test_connection_checks_auth_readiness_and_provider(settings, monkeypatch):
    import time

    monkeypatch.setenv("SHAR_AUTH_TOKENS", json.dumps({"app-token": "local"}))
    app = create_app(settings.model_copy(update={"auth_mode": "token"}))
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    assert client.get("/connection").status_code == 401
    headers = {"Authorization": "Bearer app-token"}
    assert client.get("/connection", headers=headers).status_code == 503
    app.state.store.put("worker-heartbeat", str(time.time()).encode())
    response = client.get("/connection", headers=headers)
    assert response.status_code == 200
    assert response.json() == {"status": "ready", "provider": "mock", "model": "fixture-v1"}


def test_connection_exposes_saved_quota_and_reset_requires_auth(settings, monkeypatch):
    import time

    from shar.errors import SharError
    from shar.inference.multimodal.limits import ProviderCooldown

    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setenv("SHAR_AUTH_TOKENS", json.dumps({"app-token": "local"}))
    settings = settings.model_copy(
        update={
            "provider": "gemini",
            "model": "test-model",
            "auth_mode": "token",
        }
    )
    app = create_app(settings)
    app.state.store.put("worker-heartbeat", str(time.time()).encode())
    error = SharError("PROVIDER_DAILY_QUOTA", "Daily quota exhausted.", False, 429)
    error.retry_after = 53000
    ProviderCooldown(settings).record(error)
    client = TestClient(app)
    headers = {"Authorization": "Bearer app-token"}
    response = client.get("/connection", headers=headers)
    assert response.status_code == 429
    assert response.json()["error"]["code"] == error.code
    assert response.json()["error"]["retry_at"] > time.time() + 52000
    assert client.post("/connection/recheck").status_code == 401
    assert client.post("/connection/recheck", headers=headers).status_code == 200
    assert client.get("/connection", headers=headers).status_code == 200
