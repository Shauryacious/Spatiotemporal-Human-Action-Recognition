from pathlib import Path

from shar.config import Settings
from shar.schemas import FrameRecord, VideoMetadata
from shar.video.preprocess import prepare
from shar.video.reader import decoded


def windows(duration: float, width: float, overlap: float):
    if duration <= 0 or width <= 0 or not 0 <= overlap < width:
        raise ValueError("invalid window geometry")
    start = 0.0
    while True:
        end = min(duration, start + width)
        yield start, end
        if end >= duration:
            break
        start += width - overlap


def targets(start: float, end: float, fps: float, cap: int) -> list[float]:
    if not 0 <= start < end or fps <= 0 or cap < 2:
        raise ValueError("invalid sampling domain")
    count = min(cap, max(1, int((end - start) * fps + 0.999999)))
    step = max(1 / fps, (end - start) / count)
    return [start + i * step for i in range(count)]


def sample(
    path: Path, metadata: VideoMetadata, start: float, end: float, fps: float, settings: Settings
) -> list[FrameRecord]:
    motion_slots = (
        min(settings.motion_extra_frames, settings.max_frames // 2)
        if settings.motion_sampling
        else 0
    )
    requested = targets(start, end, fps, settings.max_frames - motion_slots)
    selected = []
    seen = set()
    previous = None
    cursor = 0
    # Full presentation-order decode keeps exact indices on VFR sources. The reader also
    # exposes seek-forward decoding for callers which only need stable PTS identities.
    iterator = decoded(
        path,
        metadata,
        max(0.0, start - 1 / fps) if path.with_suffix(".index.sqlite").exists() else 0.0,
    )
    try:
        for current in iterator:
            while cursor < len(requested) and current[2] >= requested[cursor]:
                options = [f for f in (previous, current) if f is not None and start <= f[2] < end]
                if options:
                    chosen = min(options, key=lambda f: (abs(f[2] - requested[cursor]), -f[2]))
                    if chosen[1] not in seen:
                        selected.append(
                            prepare(
                                chosen[3],
                                chosen[0],
                                chosen[1],
                                chosen[2],
                                settings.longest_edge,
                                settings.jpeg_quality,
                                settings.grayscale,
                            )
                        )
                        seen.add(chosen[1])
                cursor += 1
            previous = current
            if cursor == len(requested) or current[2] >= end:
                break
        if (
            cursor < len(requested)
            and previous
            and start <= previous[2] < end
            and previous[1] not in seen
        ):
            selected.append(
                prepare(
                    previous[3],
                    previous[0],
                    previous[1],
                    previous[2],
                    settings.longest_edge,
                    settings.jpeg_quality,
                    settings.grayscale,
                )
            )
    finally:
        iterator.close()
    if motion_slots:
        import heapq

        import numpy as np

        motion: list[tuple[float, int, FrameRecord]] = []
        previous_gray = None
        iterator = decoded(
            path,
            metadata,
            max(0.0, start - 1 / fps) if path.with_suffix(".index.sqlite").exists() else 0.0,
        )
        try:
            for index, pts, timestamp, image in iterator:
                if timestamp >= end:
                    break
                gray = np.asarray(image.convert("L").resize((32, 32)), dtype=float) / 255
                score = (
                    float(np.abs(gray - previous_gray).mean()) if previous_gray is not None else 0.0
                )
                previous_gray = gray
                if timestamp < start or pts in seen or score < settings.motion_threshold:
                    continue
                if len(motion) < motion_slots or score > motion[0][0]:
                    frame = prepare(
                        image,
                        index,
                        pts,
                        timestamp,
                        settings.longest_edge,
                        settings.jpeg_quality,
                        settings.grayscale,
                    )
                    item = (score, pts, frame)
                    if len(motion) < motion_slots:
                        heapq.heappush(motion, item)
                    else:
                        heapq.heapreplace(motion, item)
        finally:
            iterator.close()
        selected.extend(item[2] for item in motion)
    return sorted(selected, key=lambda f: f.timestamp_sec)


def batches(frames: list[FrameRecord], byte_cap: int, frame_cap: int):
    batch: list[FrameRecord] = []
    size = 0
    for frame in frames:
        if len(frame.jpeg) > byte_cap:
            raise ValueError("one frame exceeds request byte budget")
        if batch and (size + len(frame.jpeg) > byte_cap or len(batch) >= frame_cap):
            yield batch
            batch, size = [], 0
        batch.append(frame)
        size += len(frame.jpeg)
    if batch:
        yield batch
