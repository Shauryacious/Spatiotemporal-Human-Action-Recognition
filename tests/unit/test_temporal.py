import pytest
from conftest import frame

from shar.inference.temporal.boundaries import brackets
from shar.inference.temporal.merge import nms, stitch, tiou
from shar.schemas import ActionPrediction, Presence


def pred(start, end, **kwargs):
    return ActionPrediction(label="walk", start_sec=float(start), end_sec=float(end), **kwargs)


def test_iou_nms_actors():
    assert tiou((0, 2), (1, 3)) == pytest.approx(1 / 3)
    a = pred(0, 2, actor_ids=["a"], confidence=0.9)
    b = pred(0, 2, actor_ids=["b"], confidence=0.8)
    c = pred(0, 2, actor_ids=["a"], confidence=0.7)
    assert len(nms([a, b, c])) == 2
    assert len(nms([pred(0, 2), pred(0, 2)])) == 2


def test_stitch_needs_continuity():
    a = pred(0, 3, continues_after=True, evidence_frame_ids=["f2"])
    b = pred(2, 4, continues_before=True, evidence_frame_ids=["f2"])
    assert stitch([a, b])[0].end_sec == 4
    b.continues_before = False
    assert len(stitch([a, b])) == 2


def test_local_brackets_multiple_occurrences():
    frames = [frame(i, float(i)) for i in range(6)]
    presence = [
        Presence(frame_id=f"f{i}", state=s)
        for i, s in enumerate(["absent", "present", "absent", "absent", "present", "absent"])
    ]
    start, end = brackets(pred(3.5, 4.5), presence, frames)
    assert (start.lower, start.upper) == (3.0, 4.0)
    assert (end.lower, end.upper) == (4.0, 5.0)
