import asyncio
import collections
import threading
import time
from dataclasses import dataclass

from shar.errors import SharError
from shar.inference.multimodal.base import ModelRequest
from shar.inference.pipeline import Pipeline
from shar.inference.temporal.merge import same_actor, tiou
from shar.schemas import FrameRecord, uid
from shar.video.preprocess import prepare


@dataclass
class StreamFrame:
    sequence: int
    epoch: int
    arrival: float
    frame: FrameRecord
    source_pts: float | None = None


class FrameRing:
    def __init__(self, capacity: int):
        self.frames: collections.deque[StreamFrame] = collections.deque(maxlen=capacity)
        self.lock = threading.Lock()
        self.dropped = 0

    def append(self, frame):
        with self.lock:
            if len(self.frames) == self.frames.maxlen:
                self.dropped += 1
            self.frames.append(frame)

    def snapshot(self):
        with self.lock:
            return list(self.frames)


class LatestWindow:
    def __init__(self):
        self.queue: asyncio.Queue[list[StreamFrame]] = asyncio.Queue(maxsize=1)
        self.dropped: collections.deque[tuple[float, float]] = collections.deque(maxlen=1000)
        self.total_dropped = 0

    def offer(self, window):
        if self.queue.full():
            previous = self.queue.get_nowait()
            self.queue.task_done()
            self.total_dropped += 1
            self.dropped.append((previous[0].frame.timestamp_sec, previous[-1].frame.timestamp_sec))
        self.queue.put_nowait(window)


class Capture:
    def __init__(self, source, settings, ring, factory=None):
        self.source, self.settings, self.ring, self.factory = source, settings, ring, factory
        self.stop_event = threading.Event()
        self.thread = None
        self.reconnects = 0
        self.error = None

    def start(self):
        self.thread = threading.Thread(target=self.run, daemon=True, name="shar-capture")
        self.thread.start()

    def run(self):
        from PIL import Image

        if not self.factory:
            import cv2

        epoch, sequence, origin, delay = 0, 0, time.monotonic(), 0.25
        while not self.stop_event.is_set():
            if self.factory:
                cap = self.factory(self.source)
            elif isinstance(self.source, str):
                cap = cv2.VideoCapture(
                    self.source,
                    cv2.CAP_FFMPEG,
                    [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000],
                )
            else:
                cap = cv2.VideoCapture(self.source)
            try:
                last_sample = -1.0
                while not self.stop_event.is_set():
                    success, rgb = cap.read()
                    now = time.monotonic()
                    if not success:
                        break
                    if now - last_sample < 1 / self.settings.coarse_fps:
                        continue
                    last_sample = now
                    image = Image.fromarray(rgb[:, :, ::-1])
                    frame = prepare(
                        image,
                        sequence,
                        sequence,
                        now - origin,
                        self.settings.longest_edge,
                        self.settings.jpeg_quality,
                    )
                    frame.frame_id = f"epoch_{epoch}_seq_{sequence}"
                    self.ring.append(StreamFrame(sequence, epoch, now, frame, cap.get(0) / 1000))
                    sequence += 1
                    delay = 0.25
            except Exception:
                self.error = "CAPTURE_ERROR"
            finally:
                cap.release()
            epoch += 1
            self.reconnects += 1
            self.stop_event.wait(delay)
            delay = min(delay * 2, 10.0)

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=6)
            if self.thread.is_alive():
                raise SharError(
                    "CAPTURE_STOP_TIMEOUT", "Capture backend did not honor its read timeout"
                )


def admit(settings):
    per_window = 1 + settings.max_candidates * (
        (1 if settings.enable_spatial else 0)
        + settings.max_rounds * (2 if settings.enable_spatial else 1)
        if settings.enable_refinement
        else (1 if settings.enable_spatial else 0)
    )
    demand = settings.stream_count * 60 / settings.stream_stride_sec * per_window * 3
    if settings.provider != "mock":
        daily = demand * 1440
        frames = min(settings.max_frames, int(settings.stream_window_sec * settings.coarse_fps + 1))
        token_demand = demand * (
            frames * settings.estimated_tokens_per_frame + settings.output_tokens
        )
        if demand > settings.rpm or daily > settings.rpd or token_demand > settings.tpm:
            raise SharError(
                "STREAM_QUOTA", "Stream profile exceeds configured worst-case account quota"
            )
    return demand


class EventTracker:
    def __init__(self, quiet_sec=8.0, max_events=100):
        self.active = {}
        self.quiet_sec, self.max_events = quiet_sec, max_events

    def update(self, predictions, epoch, now, gap=False):
        changes = []
        for prediction in predictions:
            match = next(
                (
                    event
                    for event in self.active.values()
                    if event["epoch"] == epoch
                    and event["prediction"].label == prediction.label
                    and tiou(event["prediction"], prediction) > 0
                    and same_actor(event["prediction"], prediction)
                    and event["prediction"].continues_after
                    and prediction.continues_before
                    and not gap
                ),
                None,
            )
            if match:
                match["revision"] += 1
                match["state"] = "updated"
                match["prediction"] = prediction.model_copy(
                    update={
                        "start_sec": min(prediction.start_sec, match["prediction"].start_sec),
                        "start_boundary": None,
                    }
                )
                match["last_seen"] = now
            else:
                if len(self.active) >= self.max_events:
                    oldest = min(self.active.values(), key=lambda e: e["last_seen"])
                    changes.append(self.finish(oldest, True))
                match = {
                    "event_id": uid(),
                    "revision": 1,
                    "state": "provisional",
                    "prediction": prediction,
                    "epoch": epoch,
                    "last_seen": now,
                    "uncertain_end": gap,
                }
                self.active[match["event_id"]] = match
            changes.append(self.serialize(match))
        for event in list(self.active.values()):
            if gap:
                event["uncertain_end"] = True
            if event["epoch"] != epoch or now - event["last_seen"] >= self.quiet_sec:
                changes.append(self.finish(event, gap or event["epoch"] != epoch))
        return changes

    def serialize(self, event):
        return {**event, "prediction": event["prediction"].model_dump(mode="json")}

    def finish(self, event, uncertain):
        event.update(
            state="finalized",
            revision=event["revision"] + 1,
            uncertain_end=uncertain or event["uncertain_end"],
        )
        self.active.pop(event["event_id"], None)
        return self.serialize(event)


async def stream(
    settings,
    source_name,
    emit,
    duration_sec=None,
    capture_factory=None,
    model=None,
    query="Describe observable human actions",
    preview=None,
):
    admit(settings)
    if source_name not in settings.stream_sources:
        raise SharError("STREAM_SOURCE", "Source is not in the operator-controlled registry")
    ring = FrameRing(max(2, int(settings.stream_ring_sec * settings.coarse_fps) + 2))
    capture = Capture(settings.stream_sources[source_name], settings, ring, capture_factory)
    pending, tracker = (
        LatestWindow(),
        EventTracker(settings.stream_window_sec, settings.max_candidates),
    )
    pipeline = Pipeline(settings, model=model)
    pipeline.scope, pipeline.video_hash = "stream", uid()
    started, last_end, last_drop = time.monotonic(), -1.0, 0
    capture.start()

    last_preview = -1

    async def schedule():
        nonlocal last_end, last_preview
        while duration_sec is None or time.monotonic() - started < duration_sec:
            snapshot = ring.snapshot()
            if snapshot:
                if preview and snapshot[-1].sequence != last_preview:
                    await asyncio.to_thread(preview, snapshot[-1].frame.jpeg)
                    last_preview = snapshot[-1].sequence
                epoch = snapshot[-1].epoch
                same_epoch = [f for f in snapshot if f.epoch == epoch]
                end = same_epoch[-1].frame.timestamp_sec
                selected = [
                    f
                    for f in same_epoch
                    if f.frame.timestamp_sec >= end - settings.stream_window_sec
                ]
                if (
                    selected
                    and end - selected[0].frame.timestamp_sec
                    >= settings.stream_window_sec - 1 / settings.coarse_fps
                    and end - last_end >= settings.stream_stride_sec
                ):
                    pending.offer(selected)
                    last_end = end
            await asyncio.sleep(0.05)

    producer = asyncio.create_task(schedule())
    failed_since_last = False
    try:
        while not producer.done() or not pending.queue.empty():
            try:
                window = await asyncio.wait_for(pending.queue.get(), 0.2)
            except TimeoutError:
                continue
            gap = pending.total_dropped > last_drop or failed_since_last
            last_drop = pending.total_dropped
            frames = [f.frame for f in window][-settings.max_frames :]
            try:
                response = await pipeline.infer(
                    ModelRequest(
                        task="temporal",
                        query=query,
                        frames=frames,
                        start_sec=frames[0].timestamp_sec,
                        end_sec=frames[-1].timestamp_sec + 1 / settings.coarse_fps,
                    )
                )
                from shar.inference.refinement import refine
                from shar.inference.spatial.localizer import MLLMLocalizer

                predictions = response.predictions[: settings.max_candidates]
                localizer = (
                    MLLMLocalizer(pipeline.infer, query) if settings.enable_spatial else None
                )
                analyzed = []
                for prediction in predictions:
                    actor_tracks = await localizer.localize(prediction, frames) if localizer else []
                    prediction = prediction.model_copy(
                        update={
                            "actor_ids": [t.actor_id for t in actor_tracks],
                            "spatial_status": "partial"
                            if actor_tracks
                            else ("unlocalized" if localizer else "disabled"),
                        }
                    )
                    if settings.enable_refinement:

                        async def current_frames(current, iteration):
                            return [
                                f.frame
                                for f in ring.snapshot()
                                if f.epoch == window[-1].epoch
                                and current.start_sec - 1
                                <= f.frame.timestamp_sec
                                <= current.end_sec + 1
                            ][-settings.max_frames :]

                        prediction, actor_tracks, revisions, reason = await refine(
                            prediction,
                            actor_tracks,
                            current_frames,
                            pipeline.infer,
                            localizer,
                            settings,
                            pipeline.budget,
                        )
                        emit({"type": "refinement", "stop_reason": reason, "history": revisions})
                    if actor_tracks:
                        emit(
                            {
                                "type": "tracks",
                                "tracks": [t.model_dump(mode="json") for t in actor_tracks],
                            }
                        )
                    analyzed.append(prediction)
                changes = tracker.update(
                    analyzed,
                    window[-1].epoch,
                    frames[-1].timestamp_sec,
                    gap,
                )
                failed_since_last = False
                for change in changes:
                    emit({"type": "event", **change})
                emit(
                    {
                        "type": "window",
                        "start_sec": frames[0].timestamp_sec,
                        "end_sec": frames[-1].timestamp_sec,
                        "frame_age_sec": time.monotonic() - window[-1].arrival,
                        "dropped_windows": pending.total_dropped,
                        "gap_before": gap,
                        "reconnects": capture.reconnects,
                        "epoch": window[-1].epoch,
                        "calls": pipeline.usage.model_calls,
                    }
                )
            except SharError as exc:
                failed_since_last = True
                emit(
                    {
                        "type": "failed_window",
                        "code": exc.code,
                        "start_sec": frames[0].timestamp_sec,
                        "end_sec": frames[-1].timestamp_sec,
                    }
                )
                if exc.code in {"CALL_BUDGET", "DEADLINE", "DAILY_QUOTA"}:
                    break
            finally:
                pending.queue.task_done()
        if producer.done():
            await producer
    finally:
        producer.cancel()
        import contextlib

        with contextlib.suppress(asyncio.CancelledError):
            await producer
        try:
            await asyncio.to_thread(capture.stop)
        finally:
            await pipeline.model.close()
        for event in list(tracker.active.values()):
            emit({"type": "event", **tracker.finish(event, True)})
        emit(
            {
                "type": "session_end",
                "dropped_intervals": list(pending.dropped),
                "dropped_windows": pending.total_dropped,
                "reconnects": capture.reconnects,
                "ring_frames": len(ring.snapshot()),
                "elapsed_sec": time.monotonic() - started,
            }
        )
