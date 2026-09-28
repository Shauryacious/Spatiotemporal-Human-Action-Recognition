import sys
from pathlib import Path

import pytest

from shar.config import Settings
from shar.schemas import FrameRecord, Transform

sys.path.insert(0, str(Path(__file__).parents[1]))


@pytest.fixture
def settings(tmp_path):
    return Settings(storage_root=tmp_path / "runtime", enable_refinement=False)


@pytest.fixture(scope="session")
def media(tmp_path_factory):
    from scripts.make_fixtures import make

    destination = tmp_path_factory.mktemp("media")
    make(destination)
    return destination


@pytest.fixture
def asset(media, settings):
    from shar.video.ingest import ingest_path

    return ingest_path(media / "cfr.mp4", "local", settings)


def frame(index=0, timestamp=0.0):
    return FrameRecord(
        frame_id=f"f{index}",
        frame_index=index,
        pts=index,
        timestamp_sec=timestamp,
        transform=Transform(
            source_width=160, source_height=96, prepared_width=160, prepared_height=96
        ),
        image_hash=str(index),
        jpeg=b"jpeg",
    )
