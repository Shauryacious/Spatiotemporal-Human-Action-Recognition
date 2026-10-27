from shar.storage.repository import Repository
from shar.workers.runner import Worker


async def test_restart_and_cancel(settings, asset):
    repo = Repository(settings)
    repo.add_video(asset)
    job = repo.submit(
        "local", {"video_id": asset.id, "prompt": "walk"}, settings.model_dump(mode="json")
    )
    assert await Worker(settings).once()
    assert Repository(settings).job(job)["status"] == "succeeded"
    activity = repo.job_activity(job, "local")
    stages = [entry["details"].get("stage") for entry in activity]
    assert "analysis" in stages
    assert "finalizing" in stages
    assert any(entry["details"].get("total") for entry in activity)
    assert not await Worker(settings).once()
    queued = repo.submit(
        "local", {"video_id": asset.id, "prompt": "walk"}, settings.model_dump(mode="json")
    )
    repo.cancel(queued, "local")
    assert not await Worker(settings).once()
    assert repo.job(queued)["status"] == "cancelled"


async def test_cancel_during_inference_blocks_publication(settings, asset):
    import asyncio

    from shar.inference.multimodal.mock import MockModel

    repo = Repository(settings)
    repo.add_video(asset)
    job = repo.submit(
        "local", {"video_id": asset.id, "prompt": "walk"}, settings.model_dump(mode="json")
    )
    worker = Worker(settings, model_factory=lambda _: MockModel(delay=2.0))
    import time

    started = time.monotonic()
    task = asyncio.create_task(worker.once())
    await asyncio.sleep(0.1)
    repo.cancel(job, "local")
    await task
    row = repo.job(job)
    assert row["status"] == "cancelled" and row["result_uri"] is None
    assert time.monotonic() - started < 1.0


async def test_daily_quota_fails_once_and_api_explains_why(settings, asset):
    from fastapi.testclient import TestClient

    from shar.api.app import create_app
    from shar.errors import SharError
    from shar.inference.multimodal.mock import MockModel

    repo = Repository(settings)
    repo.add_video(asset)
    job = repo.submit(
        "local", {"video_id": asset.id, "prompt": "walk"}, settings.model_dump(mode="json")
    )
    error = SharError("PROVIDER_DAILY_QUOTA", "Gemini's daily quota is exhausted.", False, 429)
    error.retry_after = 53008
    model = MockModel(responses=[error])
    worker = Worker(settings, model_factory=lambda _: model)
    assert await worker.once()
    assert not await worker.once()
    row = repo.job(job)
    assert row["status"] == "failed"
    assert row["attempt"] == row["calls"] == model.calls == 1
    assert row["next_attempt_at"] == 0
    with TestClient(create_app(settings), client=("127.0.0.1", 123)) as client:
        response = client.get(f"/jobs/{job}")
    assert response.status_code == 200
    status = response.json()
    assert status["error_message"] == error.message
    assert status["retry_at"] >= row["completed_at"] + 53008
    assert status["activity"][-1]["details"]["stage"] == "analysis"


async def test_provider_failure_is_terminal_until_user_approval(settings, asset):
    from shar.errors import SharError
    from shar.inference.multimodal.mock import MockModel

    repo = Repository(settings)
    repo.add_video(asset)
    job = repo.submit(
        "local", {"video_id": asset.id, "prompt": "walk"}, settings.model_dump(mode="json")
    )
    model = MockModel(responses=[SharError("PROVIDER_TRANSIENT", "busy", True)] * 3)
    worker = Worker(settings, model_factory=lambda _: model)
    assert await worker.once()
    assert not await worker.once()
    row = repo.job(job)
    assert row["status"] == "failed" and row["attempt"] == 1 and row["calls"] == 1


async def test_repeated_video_query_reuses_cache_across_job_deadlines(settings, asset):
    import json

    from shar.inference.multimodal.mock import MockModel

    repo = Repository(settings)
    repo.add_video(asset)
    model = MockModel()
    worker = Worker(settings, model_factory=lambda _: model)
    reports = []
    for timeout in (1800, 1700):
        config = settings.model_copy(update={"job_timeout_sec": timeout})
        job = repo.submit(
            "local", {"video_id": asset.id, "prompt": "walk"}, config.model_dump(mode="json")
        )
        await worker.once()
        row = repo.job(job)
        assert row["status"] == "succeeded"
        reports.append(json.loads(worker.store.get(row["result_uri"])))
    assert model.calls == 1
    assert reports[1]["usage"]["model_calls"] == 0
    assert reports[1]["usage"]["cache_hits"] == 1
