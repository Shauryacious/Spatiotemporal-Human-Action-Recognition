import time

from conftest import frame

from shar.inference.multimodal.limits import Budget
from shar.inference.refinement import refine
from shar.schemas import ActionPrediction, ModelPayload


async def test_unsupported_change_rolls_back(settings):
    settings = settings.model_copy(update={"max_rounds": 3})
    old = ActionPrediction(label="walk", start_sec=0.0, end_sec=1.0, evidence_frame_ids=["f0"])

    async def sample(*args):
        return [frame(), frame(1, 0.5)]

    calls = []

    async def infer(request):
        calls.append(request)
        return ModelPayload(predictions=[old.model_copy(update={"end_sec": 0.8})])

    current, _, history, reason = await refine(
        old, [], sample, infer, None, settings, Budget(5, time.monotonic() + 5)
    )
    assert current == old and reason == "invalid_refinement" and len(calls) == 1
    assert not history[0]["accepted"]


def test_spatial_convergence_does_not_switch_actors_per_frame():
    from shar.inference.refinement import align_tracks, spatial_stable
    from shar.schemas import ActorTrack, BBox, SpatialDetection

    def obs(t, x):
        return SpatialDetection(
            frame_id=f"f{t}",
            frame_index=t,
            timestamp_sec=float(t),
            bbox=BBox(x=x, y=0.1, width=0.2, height=0.4),
        )

    old = [
        ActorTrack(actor_id="a", samples=[obs(0, 0.1), obs(1, 0.1)]),
        ActorTrack(actor_id="b", samples=[obs(0, 0.7), obs(1, 0.7)]),
    ]
    switched = [
        ActorTrack(samples=[obs(0, 0.1), obs(1, 0.7)]),
        ActorTrack(samples=[obs(0, 0.7), obs(1, 0.1)]),
    ]
    assert not spatial_stable(old, align_tracks(old, switched))
    same = [t.model_copy(update={"actor_id": str(i)}) for i, t in enumerate(old)]
    assert spatial_stable(old, align_tracks(old, same))
