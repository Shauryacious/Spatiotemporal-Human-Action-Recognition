import json

import pytest
from fastapi.testclient import TestClient

from shar.api.app import create_app
from shar.errors import SharError
from shar.inference.multimodal.mock import MockModel
from shar.schemas import ActionPrediction, ActorTrack, BBox, ModelPayload, SpatialDetection
from shar.workers.runner import Worker


def temporal(request):
    return ModelPayload(
        predictions=[
            ActionPrediction(
                label="walking",
                start_sec=request.start_sec,
                end_sec=request.end_sec,
                evidence_frame_ids=[f.frame_id for f in request.frames],
            )
        ]
    )


def refined(request):
    return ModelPayload(
        predictions=[
            ActionPrediction.model_validate(
                {
                    **request.context["prediction"],
                    "evidence_frame_ids": [f.frame_id for f in request.frames],
                }
            )
        ]
    )


def spatial(request):
    f = request.frames[0]
    return ModelPayload(
        tracks=[
            ActorTrack(
                samples=[
                    SpatialDetection(
                        frame_id=f.frame_id,
                        frame_index=f.frame_index,
                        timestamp_sec=f.timestamp_sec,
                        bbox=BBox(x=0.1, y=0.1, width=0.4, height=0.5),
                    )
                ]
            )
        ]
    )


async def test_failed_refinement_waits_for_approval_and_resumes_without_repeating_work(
    settings, asset
):
    settings = settings.model_copy(update={"cache_ttl_sec": 0})
    app = create_app(settings)
    app.state.repo.add_video(asset)
    client = TestClient(app)
    job = client.post(
        "/localize",
        json={
            "video_id": asset.id,
            "prompt": "walking",
            "enable_refinement": True,
            "enable_spatial": True,
            "max_model_calls": 3,
        },
    ).json()["job_id"]
    config = app.state.repo.job(job)["config"]
    assert config["max_rounds"] == config["boundary_calls"] == config["provider_max_attempts"] == 1
    assert not config["enable_boundary_search"]
    assert config["require_retry_approval"]
    model = MockModel(
        responses=[
            temporal,
            SharError("PROVIDER_TRANSIENT", "Gemini HTTP 503: temporarily unavailable.", True, 503),
        ]
    )
    worker = Worker(settings, model_factory=lambda _: model)
    assert await worker.once()
    assert not await worker.once()
    assert model.calls == 2
    status = client.get("/jobs/" + job).json()
    assert status["status"] == "partial"
    assert status["retry_decision"]["stage"] == "refinement"
    assert "HTTP 503" in status["retry_decision"]["message"]
    assert client.get("/results/" + job).json()["predictions"][0]["label"] == "walking"
    assert client.post(f"/jobs/{job}/retry", json={}).status_code == 422
    assert client.post(f"/jobs/{job}/retry", json={"approved": False}).status_code == 422
    assert model.calls == 2
    # Global cache expiry/deletion cannot force successful calls to repeat.
    for path in (settings.storage_root / "cache").glob("*.json"):
        path.unlink()
    child = client.post(f"/jobs/{job}/retry", json={"approved": True, "max_model_calls": 1}).json()[
        "job_id"
    ]
    assert client.post(f"/jobs/{job}/retry", json={"approved": True}).json()["job_id"] == child
    model.responses.append(refined)
    await worker.once()
    assert model.calls == 3
    status = client.get("/jobs/" + child).json()
    assert status["calls"] == 1
    assert status["retry_decision"]["error_code"] == "CALL_BUDGET"
    assert status["retry_decision"]["stage"] == "localization"
    assert not await worker.once()
    # A second, separate approval funds the remaining spatial call only.
    final = client.post(f"/jobs/{child}/retry", json={"approved": True}).json()["job_id"]
    model.responses.append(spatial)
    await worker.once()
    result = client.get("/results/" + final).json()
    assert result["status"] == "succeeded"
    assert result["tracks"]
    assert result["usage"]["model_calls"] == 1
    assert result["usage"]["cache_hits"] == 2
    assert model.calls == 4
    assert not await worker.once()
    # Even after completion, repeating an old approval returns its existing child.
    assert client.post(f"/jobs/{job}/retry", json={"approved": True}).json()["job_id"] == child


async def test_temporal_error_stops_after_one_call_even_with_legacy_retry_limits(settings, asset):
    app = create_app(settings)
    app.state.repo.add_video(asset)
    config = settings.model_copy(
        update={"require_retry_approval": True, "provider_max_attempts": 5}
    )
    job = app.state.repo.submit(
        "local", {"video_id": asset.id, "prompt": "walk"}, config.model_dump(mode="json")
    )
    model = MockModel(responses=[SharError("PROVIDER_TRANSIENT", "HTTP 503", True, 503)])
    worker = Worker(settings, model_factory=lambda _: model)
    await worker.once()
    assert not await worker.once()
    assert model.calls == 1
    state = TestClient(app).get("/jobs/" + job).json()
    assert state["status"] == "failed"
    assert state["retry_decision"]["can_retry"]


async def test_unsupported_boundary_proposal_is_not_replayed_on_retry(settings, asset):
    app = create_app(settings)
    app.state.repo.add_video(asset)
    client = TestClient(app)
    job = client.post(
        "/localize", json={"video_id": asset.id, "prompt": "walk", "enable_refinement": True}
    ).json()["job_id"]

    def wrong_label(request):
        payload = refined(request)
        payload.predictions[0].label = "jumping"
        return payload

    model = MockModel(responses=[temporal, wrong_label])
    worker = Worker(settings, model_factory=lambda _: model)
    await worker.once()
    decision = client.get("/jobs/" + job).json()["retry_decision"]
    assert decision["error_code"] == "INVALID_REFINEMENT"
    assert "not supported" in decision["message"]
    child = client.post(f"/jobs/{job}/retry", json={"approved": True}).json()["job_id"]
    model.responses.append(refined)
    await worker.once()
    assert client.get("/jobs/" + child).json()["status"] == "succeeded"
    assert model.calls == 3


async def test_approval_is_owner_scoped_and_respects_provider_delay(settings, asset, monkeypatch):
    monkeypatch.setenv("SHAR_AUTH_TOKENS", json.dumps({"a": "local", "b": "other"}))
    app = create_app(settings.model_copy(update={"auth_mode": "token"}))
    app.state.repo.add_video(asset)
    client = TestClient(app)
    a, b = {"Authorization": "Bearer a"}, {"Authorization": "Bearer b"}
    job = client.post("/jobs", headers=a, json={"video_id": asset.id, "prompt": "walk"}).json()[
        "job_id"
    ]
    error = SharError("PROVIDER_RATE_LIMIT", "Wait for provider capacity.", True, 429)
    error.retry_after = 60
    await Worker(settings, model_factory=lambda _: MockModel(responses=[error])).once()
    assert client.post(f"/jobs/{job}/retry", json={"approved": True}).status_code == 401
    assert client.post(f"/jobs/{job}/retry", headers=b, json={"approved": True}).status_code == 404
    assert client.post(f"/jobs/{job}/retry", headers=a, json={"approved": True}).status_code == 429


def test_worker_interruption_requires_approval_and_fences_old_worker(settings, asset):
    import time

    from sqlalchemy import update

    from shar.storage import models as m

    app = create_app(settings)
    repo = app.state.repo
    repo.add_video(asset)
    client = TestClient(app)
    job = client.post("/jobs", json={"video_id": asset.id, "prompt": "walk"}).json()["job_id"]
    claimed = repo.claim()
    repo.reserve_call(job, claimed["lease_token"], 3)
    with repo.engine.begin() as db:
        db.execute(update(m.jobs).where(m.jobs.c.id == job).values(lease_until=time.time() - 1))
    assert repo.claim() is None
    assert not repo.heartbeat(job, claimed["lease_token"])
    state = client.get("/jobs/" + job).json()
    assert state["status"] == "failed"
    assert state["retry_decision"]["error_code"] == "WORKER_INTERRUPTED"
    assert "may have been charged" in state["retry_decision"]["message"]
    assert state["calls"] == 1


async def test_concurrent_approvals_create_only_one_child(settings, asset):
    from concurrent.futures import ThreadPoolExecutor

    app = create_app(settings)
    app.state.repo.add_video(asset)
    client = TestClient(app)
    job = client.post("/jobs", json={"video_id": asset.id, "prompt": "walk"}).json()["job_id"]
    await Worker(
        settings,
        model_factory=lambda _: MockModel(
            responses=[SharError("PROVIDER_TRANSIENT", "HTTP 503", True)]
        ),
    ).once()

    def approve(_):
        with TestClient(app) as other_tab:
            return other_tab.post(f"/jobs/{job}/retry", json={"approved": True}).json()["job_id"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        children = list(pool.map(approve, range(2)))
    assert children[0] == children[1]
    assert len(app.state.repo.list_jobs("local")) == 2


@pytest.mark.parametrize("damage", ["missing", "corrupted"])
async def test_missing_saved_response_never_triggers_paid_regeneration(settings, asset, damage):
    app = create_app(settings)
    app.state.repo.add_video(asset)
    client = TestClient(app)
    job = client.post(
        "/localize", json={"video_id": asset.id, "prompt": "walk", "enable_refinement": True}
    ).json()["job_id"]
    model = MockModel(responses=[temporal, SharError("PROVIDER_TRANSIENT", "HTTP 503", True)])
    worker = Worker(settings, model_factory=lambda _: model)
    await worker.once()
    saved = settings.storage_root / "run-responses" / job
    index = json.loads((saved / "index.json").read_text())
    assert len(index) == 1
    response_path = saved / (next(iter(index)) + ".json")
    if damage == "missing":
        response_path.unlink()
    else:
        response_path.write_text("{}")
    # Even the ordinary global cache must not conceal broken saved run data.
    child = client.post(f"/jobs/{job}/retry", json={"approved": True}).json()["job_id"]
    await worker.once()
    state = client.get("/jobs/" + child).json()
    assert state["calls"] == 0
    assert model.calls == 2
    assert state["retry_decision"]["error_code"] == "SAVED_RESPONSE_UNAVAILABLE"
    assert not state["retry_decision"]["can_retry"]


async def test_saved_daily_quota_failure_becomes_retryable_after_capacity_returns(
    settings, asset, monkeypatch
):
    from shar.inference.multimodal.limits import SharedLimiter, provider_scope

    settings = settings.model_copy(update={"rpm": 10, "tpm": 1000, "rpd": 1})
    app = create_app(settings)
    app.state.repo.add_video(asset)
    client = TestClient(app)
    job = client.post(
        "/localize",
        json={"video_id": asset.id, "prompt": "walking", "enable_refinement": False},
    ).json()["job_id"]
    # Simulate an older saved failure without retry metadata.
    model = MockModel(responses=[SharError("DAILY_QUOTA", "Local quota exhausted")])
    await Worker(settings, model_factory=lambda _: model).once()
    limiter = SharedLimiter(settings.storage_root, provider_scope(settings), 10, 1000, 1)
    limiter.reserve(20)
    status = client.get(f"/jobs/{job}").json()
    assert status["retry_decision"]["can_retry"]
    assert status["retry_decision"]["retry_at"]
    assert client.post(f"/jobs/{job}/retry", json={"approved": True}).status_code == 429
    # Expiry permits explicit approval, without dispatching any inference here.
    monkeypatch.setattr(limiter.__class__, "daily_retry_after", lambda self: 0.0)
    assert client.post(f"/jobs/{job}/retry", json={"approved": True}).status_code == 202
    assert model.calls == 1
