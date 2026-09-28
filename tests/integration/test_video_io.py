import pytest

from shar.errors import SharError
from shar.video.ingest import ingest_path
from shar.video.reader import decoded, probe
from shar.video.sampling import sample


def test_cfr_vfr_rotation(media, settings):
    cfr = probe(media / "cfr.mp4", settings)
    assert cfr.duration_sec == pytest.approx(4)
    assert len(list(decoded(media / "cfr.mp4", cfr))) == 48
    vfr = probe(media / "vfr.mp4", settings)
    frames = list(decoded(media / "vfr.mp4", vfr))
    gaps = {round(b[2] - a[2], 3) for a, b in zip(frames, frames[1:])}
    assert len(gaps) > 1
    selected = sample(media / "vfr.mp4", vfr, 0.0, 4.0, 4.0, settings)
    actual = {f[1]: f[2] for f in frames}
    assert all(f.timestamp_sec == actual[f.pts] for f in selected)
    rotated = probe(media / "rotated.mp4", settings)
    assert (rotated.width, rotated.height) == (96, 160)
    assert next(decoded(media / "rotated.mp4", rotated))[3].size == (96, 160)


def test_bad_media_removed(tmp_path, settings):
    path = tmp_path / "bad.mp4"
    path.write_bytes(b"not a video")
    with pytest.raises(SharError):
        ingest_path(path, "owner", settings)
    assert not list((settings.storage_root / "videos").glob("*"))


def test_seek_index_preserves_presentation_identity(settings, asset):
    path = settings.storage_root / asset.managed_uri
    full = {f[1]: f[0] for f in decoded(path, asset.metadata)}
    late = sample(path, asset.metadata, 2.0, 4.0, 4.0, settings)
    assert all(f.frame_index == full[f.pts] for f in late)
    assert min(f.timestamp_sec for f in late) >= 2.0


def test_motion_sampling_keeps_anchors_bounded(settings, asset):
    settings = settings.model_copy(
        update={
            "motion_sampling": True,
            "motion_threshold": 0.0,
            "max_frames": 10,
            "motion_extra_frames": 4,
        }
    )
    frames = sample(
        settings.storage_root / asset.managed_uri, asset.metadata, 0.0, 4.0, 1.0, settings
    )
    assert 4 <= len(frames) <= 10
    assert frames[0].timestamp_sec == 0.0
    assert len({f.frame_id for f in frames}) == len(frames)
