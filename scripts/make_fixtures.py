"""Generate original synthetic geometry/timing media. No downloaded footage."""

import argparse
import subprocess
from pathlib import Path


def make(destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    cfr = destination / "cfr.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x96:rate=12:duration=4",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(cfr),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(cfr),
            "-vf",
            "select='if(lt(t,2),not(mod(n,2)),1)'",
            "-fps_mode",
            "vfr",
            "-c:v",
            "libx264",
            str(destination / "vfr.mp4"),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-display_rotation",
            "90",
            "-i",
            str(cfr),
            "-c",
            "copy",
            "-metadata:s:v:0",
            "rotate=90",
            str(destination / "rotated.mp4"),
        ],
        check=True,
    )
    return cfr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/fixtures"))
    print(make(parser.parse_args().output))
