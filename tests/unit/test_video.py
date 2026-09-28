import pytest
from PIL import Image

from shar.schemas import BBox, Transform
from shar.video.preprocess import invert_box, prepare
from shar.video.sampling import batches, targets, windows


def test_windows_tail():
    assert list(windows(61, 30, 5)) == [(0.0, 30.0), (25.0, 55.0), (50.0, 61)]
    assert list(windows(30, 30, 5)) == [(0.0, 30)]
    assert len(targets(0, 100, 4, 64)) == 64
    with pytest.raises(ValueError):
        list(windows(10, 5, 5))


def test_transform():
    f = prepare(Image.new("RGB", (200, 100), "red"), 0, 0, 0.0, 100)
    assert (f.transform.prepared_width, f.transform.prepared_height) == (100, 50)
    b = invert_box(
        BBox(x=0.2, y=0.2, width=0.5, height=0.5),
        Transform(
            source_width=100,
            source_height=100,
            prepared_width=50,
            prepared_height=50,
            crop_x=0.1,
            crop_y=0.1,
            crop_width=0.5,
            crop_height=0.5,
        ),
    )
    assert b.x == pytest.approx(0.2)
    assert b.width == 0.25
    assert len(list(batches([f, f, f], len(f.jpeg) * 2, 3))) == 2
