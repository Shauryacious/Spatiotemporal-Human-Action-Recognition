import pytest

from shar.evaluation.dataset import DatasetVideo, validate_splits
from shar.evaluation.metrics import average_precision, recognition_metrics, temporal_metrics
from shar.schemas import ActionPrediction


def test_ap_hand_computed():
    assert average_precision([True, False, True], 2) == pytest.approx(5 / 6)
    assert average_precision([], 1) == 0
    assert average_precision([], 0) is None
    gt = ActionPrediction(label="walk", start_sec=0.0, end_sec=2.0)
    pred = gt.model_copy(update={"confidence": 0.9})
    result = temporal_metrics([("v", pred), ("v", pred)], [("v", gt)])
    assert result["thresholds"]["0.5"]["mAP"] == 1
    assert recognition_metrics({"v": {"walk"}}, {"v": {"walk", "run"}})[
        "micro_f1"
    ] == pytest.approx(2 / 3)


def test_leakage():
    record = DatasetVideo(
        video_id="v",
        path="x",
        sha256="a",
        duration_sec=1.0,
        split="train",
        group_id="g",
        license="synthetic",
    )
    with pytest.raises(ValueError):
        validate_splits([record, record.model_copy(update={"video_id": "v2", "split": "test"})])


def test_joint_uses_one_actor_for_whole_event():
    from shar.evaluation.matching import joint_overlap
    from shar.schemas import ActorTrack, BBox, SpatialDetection

    def obs(t, x):
        return SpatialDetection(
            frame_id=str(t),
            frame_index=t,
            timestamp_sec=float(t),
            bbox=BBox(x=x, y=0.1, width=0.2, height=0.4),
        )

    gt_track = ActorTrack(actor_id="gt", samples=[obs(0, 0.1), obs(1, 0.1)])
    a = ActorTrack(actor_id="a", samples=[obs(0, 0.1), obs(1, 0.7)])
    b = ActorTrack(actor_id="b", samples=[obs(0, 0.7), obs(1, 0.1)])
    gt = ActionPrediction(label="walk", start_sec=0.0, end_sec=2.0, actor_ids=["gt"])
    pred = gt.model_copy(update={"actor_ids": ["a", "b"]})
    overlap, coverage = joint_overlap(pred, gt, {"a": a, "b": b}, {"gt": gt_track})
    assert overlap == 0.5 and coverage == 1.0
    pred.actor_ids = []
    assert joint_overlap(pred, gt, {}, {"gt": gt_track}) == (0.0, 0.0)


def test_evaluation_profile(tmp_path):
    from shar.evaluation.dataset import EvaluationSettings

    path = tmp_path / "evaluation.yaml"
    path.write_text(
        "temporal_thresholds: [0.4, 0.6]\nspatial_threshold: 0.7\naliases: {strolling: walking}\n"
    )
    spec = EvaluationSettings.load(path)
    assert spec.aliases["strolling"] == "walking" and spec.temporal_thresholds == [0.4, 0.6]
    path.write_text("temporal_thresholds: [1.1]\n")
    with pytest.raises(ValueError):
        EvaluationSettings.load(path)
