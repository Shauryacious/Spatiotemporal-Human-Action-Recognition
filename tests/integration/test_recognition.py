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
    assert report.usage.model_calls == 3


async def test_retry_budget_is_exact(settings, asset):
    import pytest

    from shar.errors import SharError

    model = MockModel(responses=[SharError("PROVIDER_TRANSIENT", "retry", True)] * 5)
    settings = settings.model_copy(update={"max_calls": 2})
    with pytest.raises(SharError):
        await Pipeline(settings, model).run(asset, "actions")
    assert model.calls == 2


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
