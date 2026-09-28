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
