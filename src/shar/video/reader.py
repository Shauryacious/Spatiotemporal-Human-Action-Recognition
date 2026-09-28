import json
import subprocess
from pathlib import Path

from shar.config import Settings
from shar.errors import SharError
from shar.schemas import VideoMetadata


def probe(path: Path, settings: Settings) -> VideoMetadata:
    import av

    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            timeout=20,
            check=True,
        )
        raw = json.loads(result.stdout)
        stream = raw["streams"][0]
        width, height = int(stream["width"]), int(stream["height"])
        if width * height > settings.max_pixels:
            raise SharError("VIDEO_DIMENSIONS", "Video dimensions exceed the configured limit")
        rotation = int(stream.get("tags", {}).get("rotate", 0))
        for side in stream.get("side_data_list", []):
            rotation = int(side.get("rotation", rotation))
        rotation %= 360
        if rotation not in {0, 90, 180, 270}:
            raise SharError("VIDEO_ROTATION", "Only orthogonal display rotation is supported")
        with av.open(str(path)) as container:
            vs = container.streams.video[0]
            first = next(container.decode(vs))
            if first.pts is None:
                raise SharError("VIDEO_TIMESTAMPS", "Video has no usable presentation timestamps")
            # Decode a real frame before accepting the asset.
            first.to_image()
            first_pts = first.pts
            fps = float(vs.average_rate) if vs.average_rate else None
            duration = (
                float(vs.duration * vs.time_base)
                if vs.duration is not None and vs.time_base is not None
                else float(raw["format"]["duration"])
            )
        if duration > settings.max_duration_sec:
            raise SharError("VIDEO_DURATION", "Video exceeds duration limit")
        return VideoMetadata(
            duration_sec=duration,
            width=height if rotation % 180 else width,
            height=width if rotation % 180 else height,
            source_width=width,
            source_height=height,
            rotation=rotation,
            time_base=stream["time_base"],
            first_pts=first_pts,
            nominal_fps=fps,
            frame_count=int(stream["nb_frames"])
            if stream.get("nb_frames", "N/A") != "N/A"
            else None,
            codec=stream.get("codec_name", "unknown"),
        )
    except SharError:
        raise
    except Exception as exc:
        raise SharError("INVALID_VIDEO", "Unable to probe and decode video", status=415) from exc


def decoded(path: Path, metadata: VideoMetadata, start: float = 0.0):
    """Seek to an earlier keyframe and yield actual normalized PTS; constant memory."""
    import sqlite3
    from contextlib import ExitStack
    from fractions import Fraction

    import av

    time_base = Fraction(metadata.time_base)
    with ExitStack() as stack:
        container = stack.enter_context(av.open(str(path)))
        index_path = path.with_suffix(".index.sqlite")
        index_db = sqlite3.connect(index_path) if index_path.exists() else None
        if index_db:
            stack.callback(index_db.close)
        stream = container.streams.video[0]
        if start > 0:
            container.seek(
                int(start / time_base) + metadata.first_pts,
                stream=stream,
                backward=True,
                any_frame=False,
            )
        previous = -1.0
        for index, frame in enumerate(container.decode(stream)):
            if frame.pts is None:
                raise SharError("VIDEO_TIMESTAMPS", "Frame has no presentation timestamp")
            timestamp = float((frame.pts - metadata.first_pts) * time_base)
            if timestamp < 0:
                continue
            if timestamp <= previous:
                raise SharError("VIDEO_TIMESTAMPS", "Presentation timestamps are not increasing")
            previous = timestamp
            image = frame.to_image()
            if metadata.rotation:
                image = image.rotate(metadata.rotation, expand=True)
            if index_db:
                row = index_db.execute(
                    "SELECT frame_index FROM frames WHERE pts=?", (frame.pts,)
                ).fetchone()
                if row is None:
                    raise SharError("VIDEO_INDEX", "Frame is missing from the presentation index")
                index = row[0]
            yield index, frame.pts, timestamp, image


def build_index(path: Path, metadata: VideoMetadata, settings: Settings):
    """Disk-backed exact PTS/presentation index. Never accumulate full index in RAM."""
    import sqlite3
    import time

    import av

    index_path = path.with_suffix(".index.sqlite")
    started = time.monotonic()
    count = 0
    try:
        with sqlite3.connect(index_path) as db:
            db.execute(
                "CREATE TABLE frames (pts INTEGER PRIMARY KEY, frame_index INTEGER NOT NULL, timestamp REAL NOT NULL, keyframe INTEGER NOT NULL)"
            )
            db.execute("CREATE INDEX frame_times ON frames(timestamp)")
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                previous = -1.0
                for count, frame in enumerate(container.decode(stream), 1):
                    if time.monotonic() - started > settings.job_timeout_sec:
                        raise SharError("DECODE_TIMEOUT", "Video indexing exceeded deadline")
                    if frame.pts is None or stream.time_base is None:
                        raise SharError("VIDEO_TIMESTAMPS", "Frame has no usable timestamp")
                    timestamp = float((frame.pts - metadata.first_pts) * stream.time_base)
                    if timestamp <= previous:
                        raise SharError("VIDEO_TIMESTAMPS", "Non-monotonic presentation timestamps")
                    if frame.width * frame.height > settings.max_pixels:
                        raise SharError("VIDEO_DIMENSIONS", "Frame exceeds pixel limit")
                    db.execute(
                        "INSERT INTO frames VALUES (?,?,?,?)",
                        (frame.pts, count - 1, timestamp, int(frame.key_frame)),
                    )
                    previous = timestamp
            db.commit()
        metadata.frame_count = count
        metadata.count_reliability = "decoded"
    except BaseException:
        index_path.unlink(missing_ok=True)
        raise


def inspect_isolated(path: Path, settings: Settings) -> VideoMetadata:
    """Probe and build the PTS index in a time/resource-bounded decoder process."""
    import os
    import sys

    limits = {
        name: getattr(settings, name)
        for name in ("max_pixels", "max_duration_sec", "job_timeout_sec")
    }
    environment = {
        name: value
        for name, value in os.environ.items()
        if name
        in {"PATH", "PYTHONPATH", "VIRTUAL_ENV", "SYSTEMROOT", "WINDIR", "PYDANTIC_DISABLE_PLUGINS"}
    }
    try:
        result = subprocess.run(
            [sys.executable, "-m", "shar.video.reader", str(path.resolve()), json.dumps(limits)],
            capture_output=True,
            timeout=min(settings.job_timeout_sec, 120),
            env=environment,
        )
        value = json.loads(result.stdout)
        if result.returncode or "error" in value:
            error = value.get("error", {})
            raise SharError(
                error.get("code", "INVALID_VIDEO"),
                error.get("message", "Video inspection failed"),
                status=415,
            )
        return VideoMetadata.model_validate_json(result.stdout)
    except (subprocess.TimeoutExpired, ValueError) as exc:
        raise SharError(
            "VIDEO_INSPECTION", "Video inspection failed or exceeded resource limits", status=415
        ) from exc


if __name__ == "__main__":
    import sys

    try:
        inspection_settings = Settings.model_validate_json(sys.argv[2])
        if sys.platform != "win32":
            import resource

            cpu_seconds = max(1, int(min(inspection_settings.job_timeout_sec, 120)))
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
            if sys.platform == "linux":
                resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
        inspection_path = Path(sys.argv[1])
        inspected = probe(inspection_path, inspection_settings)
        build_index(inspection_path, inspected, inspection_settings)
        print(inspected.model_dump_json())
    except SharError as error:
        print(json.dumps(error.public()))
        raise SystemExit(1) from error
