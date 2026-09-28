from conftest import frame

from shar.schemas import ActionPrediction
from shar.video.stream import EventTracker, FrameRing, LatestWindow, StreamFrame


def test_bounded_ring_and_latest():
    ring = FrameRing(3)
    for i in range(20):
        ring.append(StreamFrame(i, 0, float(i), frame(i, float(i))))
    assert len(ring.snapshot()) == 3 and ring.dropped == 17
    queue = LatestWindow()
    queue.offer(ring.snapshot())
    queue.offer(ring.snapshot())
    assert queue.queue.qsize() == 1 and len(queue.dropped) == 1


def test_epoch_and_gap_finalize_uncertain():
    tracker = EventTracker(quiet_sec=1)
    prediction = ActionPrediction(label="walk", start_sec=0.0, end_sec=2.0)
    event = tracker.update([prediction], 0, 0.0)[0]
    changes = tracker.update([], 1, 2.0, gap=True)
    assert changes[0]["event_id"] == event["event_id"]
    assert changes[0]["state"] == "finalized" and changes[0]["uncertain_end"]


async def test_slow_provider_keeps_capture_bounded(settings):
    import time

    import numpy as np

    from shar.inference.multimodal.mock import MockModel
    from shar.video.stream import stream

    class FakeCapture:
        def __init__(self):
            self.released = False

        def read(self):
            time.sleep(0.01)
            return True, np.zeros((48, 64, 3), dtype=np.uint8)

        def get(self, key):
            return 0.0

        def release(self):
            self.released = True

    captures = []

    def factory(source):
        cap = FakeCapture()
        captures.append(cap)
        return cap

    settings = settings.model_copy(
        update={
            "coarse_fps": 20.0,
            "stream_window_sec": 0.2,
            "stream_stride_sec": 0.05,
            "stream_ring_sec": 0.4,
            "stream_sources": {"fake": 0},
            "max_frames": 16,
        }
    )
    events = []
    await stream(
        settings,
        "fake",
        events.append,
        duration_sec=0.8,
        capture_factory=factory,
        model=MockModel(delay=0.3),
    )
    end = events[-1]
    assert end["type"] == "session_end" and end["ring_frames"] <= 10
    assert end["dropped_intervals"]
    assert all(cap.released for cap in captures)


async def test_preview_advances_independently_of_inference(settings):
    import time

    import numpy as np

    from shar.inference.multimodal.mock import MockModel
    from shar.video.stream import stream

    class Source:
        def read(self):
            time.sleep(0.01)
            return True, np.zeros((32, 32, 3), dtype=np.uint8)

        def get(self, key):
            return 0.0

        def release(self):
            pass

    settings = settings.model_copy(
        update={
            "coarse_fps": 20.0,
            "stream_window_sec": 0.2,
            "stream_stride_sec": 0.05,
            "stream_ring_sec": 0.4,
            "stream_sources": {"fake": 0},
            "max_frames": 16,
        }
    )
    previews = []
    model = MockModel(delay=0.3)
    await stream(
        settings,
        "fake",
        lambda _: None,
        duration_sec=0.8,
        capture_factory=lambda _: Source(),
        model=model,
        preview=previews.append,
    )
    assert len(previews) > model.calls and all(p.startswith(b"\xff\xd8") for p in previews)
