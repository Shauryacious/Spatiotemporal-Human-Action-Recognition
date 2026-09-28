from shar.inference.spatial.association import associate, box_iou
from shar.schemas import BBox, SpatialDetection


def det(i, x, t):
    return SpatialDetection(
        frame_id=f"f{i}",
        frame_index=i,
        timestamp_sec=float(t),
        bbox=BBox(x=x, y=0.1, width=0.2, height=0.4),
    )


def test_tracks_and_gaps():
    tracks = associate([det(0, 0.1, 0), det(1, 0.12, 0.5), det(2, 0.13, 3)])
    assert sorted(len(t.samples) for t in tracks) == [1, 2]
    assert box_iou(tracks[0].samples[0].bbox, tracks[0].samples[0].bbox) == 1


def test_ambiguous_crossing():
    tracks = associate([det(0, 0.1, 0), det(0, 0.11, 0), det(1, 0.105, 0.5)])
    assert any(t.association_status == "ambiguous" for t in tracks)
