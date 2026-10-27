from shar.inference.multimodal.mock import MockModel
from shar.inference.pipeline import Pipeline


async def test_recognition_has_no_fake_intervals(settings, asset):
    settings = settings.model_copy(update={"enable_temporal": False})
    report = await Pipeline(settings, MockModel(demo=True)).run(asset, "actions")
    assert report.recognition and not report.predictions


async def test_empty_valid_coverage(settings, asset):
    report = await Pipeline(settings, MockModel()).run(asset, "actions")
    assert report.status == "succeeded" and not report.predictions
    assert report.coverage.analyzed_intervals == [(0.0, 4.0)]


async def test_invalid_window_is_partial_not_empty(settings, asset):
    from shar.errors import SharError
    from shar.inference.multimodal.base import ModelResponse

    settings = settings.model_copy(update={"window_sec": 2.0, "overlap_sec": 0.0})
    model = MockModel(
        responses=[
            ModelResponse({}, "mock"),
            SharError("INVALID_MODEL_OUTPUT", "bad"),
            SharError("INVALID_MODEL_OUTPUT", "bad"),
        ]
    )
    report = await Pipeline(settings, model).run(asset, "actions")
    assert report.status == "partial"
    assert report.coverage.failed_intervals == [(2.0, 4.0)]
    assert report.usage.model_calls == 2
    assert report.provenance["failure"]["error_code"] == "INVALID_MODEL_OUTPUT"


async def test_failure_never_uses_remaining_budget_without_approval(settings, asset):
    import pytest

    from shar.errors import SharError

    model = MockModel(responses=[SharError("PROVIDER_TRANSIENT", "retry", True)] * 5)
    settings = settings.model_copy(update={"max_calls": 2})
    with pytest.raises(SharError):
        await Pipeline(settings, model).run(asset, "actions")
    assert model.calls == 1


async def test_complete_temporal_spatial_refinement(settings, asset):
    from shar.schemas import (
        ActionPrediction,
        ActorTrack,
        BBox,
        ModelPayload,
        Presence,
        SpatialDetection,
    )

    def scripted(request):
        if request.task == "spatial":
            return ModelPayload(
                tracks=[
                    ActorTrack(
                        samples=[
                            SpatialDetection(
                                frame_id=f.frame_id,
                                frame_index=f.frame_index,
                                timestamp_sec=f.timestamp_sec,
                                bbox=BBox(x=0.1, y=0.1, width=0.2, height=0.4),
                            )
                            for f in request.frames
                        ]
                    )
                ]
            )
        inside = [f.frame_id for f in request.frames if 1.0 <= f.timestamp_sec < 3.0]
        prediction = ActionPrediction(
            label="walking", start_sec=1.0, end_sec=3.0, confidence=0.8, evidence_frame_ids=inside
        )
        presence = [
            Presence(
                frame_id=f.frame_id, state="present" if 1.0 <= f.timestamp_sec < 3.0 else "absent"
            )
            for f in request.frames
        ]
        return ModelPayload(predictions=[prediction], presence=presence)

    model = MockModel(responses=[scripted] * 30)
    settings = settings.model_copy(update={"enable_spatial": True, "enable_refinement": True})
    report = await Pipeline(settings, model).run(asset, "walking")
    assert report.predictions and report.tracks
    assert report.predictions[0].actor_ids == [report.tracks[0].actor_id]
    assert report.predictions[0].start_boundary is not None
    assert report.usage.model_calls <= settings.max_calls
    assert any(r["stop_reason"] in {"converged", "no_new_evidence"} for r in report.revisions)


async def test_model_version_drift_is_not_silent(settings, asset):
    import pytest

    from shar.errors import SharError
    from shar.inference.multimodal.base import ModelResponse

    settings = settings.model_copy(update={"window_sec": 2.0, "overlap_sec": 0.0})
    model = MockModel(
        responses=[ModelResponse({}, "version-one"), ModelResponse({}, "version-two")]
    )
    with pytest.raises(SharError, match="version changed"):
        await Pipeline(settings, model).run(asset, "actions")


async def test_spatial_samples_cover_action_interior(settings, asset):
    from shar.schemas import ActionPrediction, ModelPayload

    spatial_times = []

    def scripted(request):
        if request.task == "spatial":
            spatial_times.extend(f.timestamp_sec for f in request.frames)
            return ModelPayload()
        return ModelPayload(
            predictions=[
                ActionPrediction(
                    label="walking",
                    start_sec=0.0,
                    end_sec=4.0,
                    evidence_frame_ids=[f.frame_id for f in request.frames],
                )
            ]
        )

    settings = settings.model_copy(
        update={
            "enable_spatial": True,
            "enable_boundary_search": False,
        }
    )
    report = await Pipeline(settings, MockModel(responses=[scripted] * 2)).run(asset, "walking")
    assert any(1.5 <= timestamp <= 2.5 for timestamp in spatial_times)
    assert all(0 <= timestamp < 4 for timestamp in spatial_times)
    assert report.predictions[0].spatial_status == "unlocalized"


async def test_provider_failure_requires_approval_without_retrying_job(settings, asset):
    import pytest

    from shar.errors import SharError

    model = MockModel(responses=[SharError("PROVIDER_TRANSIENT", "retry", True)] * 3)
    activity = []
    with pytest.raises(SharError) as caught:
        await Pipeline(settings, model, checkpoint=activity.append).run(asset, "actions")
    assert model.calls == 1
    assert caught.value.code == "PROVIDER_TRANSIENT"
    assert not caught.value.retryable
    assert any(item.get("failure", {}).get("can_retry") for item in activity)


async def test_provider_cooldown_longer_than_job_deadline_fails_immediately(settings, asset):
    import time

    import pytest

    from shar.errors import SharError

    error = SharError("PROVIDER_RATE_LIMIT", "Rate limit reached.", True, 429)
    error.retry_after = 53008
    model = MockModel(responses=[error])
    started = time.monotonic()
    with pytest.raises(SharError) as caught:
        await Pipeline(settings, model).run(asset, "actions")
    assert model.calls == 1
    assert not caught.value.retryable
    assert caught.value.retry_after == 53008
    assert caught.value.message == "Rate limit reached."
    assert time.monotonic() - started < 3


async def test_cancel_during_explicitly_enabled_automatic_retry(settings, asset):
    settings = settings.model_copy(update={"require_retry_approval": False})
    import asyncio
    import time

    import pytest

    from shar.errors import Cancelled, SharError

    error = SharError("PROVIDER_RATE_LIMIT", "Rate limit reached.", True, 429)
    error.retry_after = 30
    retrying = asyncio.Event()
    cancelled = False
    model = MockModel(responses=[error])
    pipeline = Pipeline(
        settings,
        model,
        cancelled=lambda: cancelled,
        checkpoint=lambda details: retrying.set() if details.get("retry_after_sec") else None,
    )
    task = asyncio.create_task(pipeline.run(asset, "actions"))
    await asyncio.wait_for(retrying.wait(), 5)
    started = time.monotonic()
    cancelled = True
    with pytest.raises(Cancelled):
        await asyncio.wait_for(task, 1)
    assert time.monotonic() - started < 1
    assert model.calls == 1


async def test_prevented_request_does_not_count_as_model_call(settings, asset):
    from unittest.mock import Mock

    import pytest

    from shar.errors import SharError

    model = MockModel()
    durable_reserve = Mock()
    pipeline = Pipeline(settings, model, reserve=durable_reserve)
    pipeline.limiter = Mock()
    pipeline.limiter.reserve.side_effect = SharError("DAILY_QUOTA", "Local quota exhausted")
    with pytest.raises(SharError):
        await pipeline.run(asset, "actions")
    assert model.calls == pipeline.usage.model_calls == pipeline.budget.calls == 0
    durable_reserve.assert_not_called()


async def test_daily_quota_blocks_following_jobs_without_calling_provider(settings, asset):
    import pytest

    from shar.errors import SharError

    settings = settings.model_copy(
        update={
            "provider": "gemini",
            "model": "test-model",
            "rpm": 100,
            "tpm": 100000,
            "rpd": 100,
        }
    )
    error = SharError("PROVIDER_DAILY_QUOTA", "Daily quota exhausted.", False, 429)
    error.retry_after = 53000
    first = MockModel(responses=[error])
    with pytest.raises(SharError):
        await Pipeline(settings, first).run(asset, "actions")
    assert first.calls == 1
    second = MockModel()
    pipeline = Pipeline(settings, second)
    with pytest.raises(SharError) as caught:
        await pipeline.run(asset, "actions")
    assert caught.value.code == "PROVIDER_DAILY_QUOTA"
    assert second.calls == pipeline.usage.model_calls == 0


async def test_one_call_budget_does_not_retry_or_hide_provider_error(settings, asset):
    import pytest

    from shar.errors import SharError

    settings = settings.model_copy(update={"max_calls": 1})
    model = MockModel(responses=[SharError("PROVIDER_TRANSIENT", "Provider busy", True)])
    activity = []
    with pytest.raises(SharError) as caught:
        await Pipeline(settings, model, checkpoint=activity.append).run(asset, "actions")
    assert caught.value.code == "PROVIDER_TRANSIENT"
    assert not caught.value.retryable
    assert model.calls == 1
    assert not any("Retrying" in entry["message"] for entry in activity)
