"""Bounded, offline stream soak; never opens a camera or calls a hosted model."""

import argparse
import asyncio
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

from shar.config import Settings
from shar.inference.multimodal.mock import MockModel
from shar.video.stream import stream


async def profile(duration, output):
    class SyntheticCapture:
        def read(self):
            time.sleep(0.01)
            return True, np.zeros((96, 160, 3), dtype=np.uint8)

        def get(self, key):
            return 0.0

        def release(self):
            pass

    settings = Settings(
        enable_refinement=False,
        coarse_fps=20.0,
        stream_window_sec=0.2,
        stream_stride_sec=0.05,
        stream_ring_sec=0.4,
        stream_sources={"synthetic": 0},
        max_frames=16,
        max_calls=1000,
        storage_root=output.parent / "profile-runtime",
    )
    windows = []
    session = {}

    def emit(event):
        if event["type"] == "window":
            windows.append(event["frame_age_sec"])
        if event["type"] == "session_end":
            session.update(event)

    await stream(
        settings,
        "synthetic",
        emit,
        duration_sec=duration,
        capture_factory=lambda _: SyntheticCapture(),
        model=MockModel(delay=0.25),
    )
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {
        "kind": "offline synthetic slow-provider soak",
        "duration_sec": duration,
        "max_ring_capacity": 10,
        "final_ring_frames": session["ring_frames"],
        "processed_windows": len(windows),
        "dropped_windows": session["dropped_windows"],
        "frame_age_p50_sec": float(np.percentile(windows, 50)) if windows else None,
        "frame_age_p95_sec": float(np.percentile(windows, 95)) if windows else None,
        "process_peak_rss_bytes": peak if sys.platform == "darwin" else peak * 1024,
        "note": "Process peak includes Python/imports; this is not a real camera or provider latency benchmark.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--output", type=Path, default=Path("outputs/soak.json"))
    args = parser.parse_args()
    asyncio.run(profile(args.duration, args.output))
