from shar.inference.multimodal.base import ModelRequest
from shar.inference.spatial.association import box_iou
from shar.inference.temporal.boundaries import supported_update


def align_tracks(old_tracks, new_tracks):
    """Match entire observed tracks, never select a different actor per frame."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    if not old_tracks or not new_tracks:
        return new_tracks
    scores = np.zeros((len(old_tracks), len(new_tracks)))
    for i, old in enumerate(old_tracks):
        for j, new in enumerate(new_tracks):
            if "ambiguous" in {old.association_status, new.association_status}:
                continue
            a = {s.frame_id: s.bbox for s in old.samples}
            b = {s.frame_id: s.bbox for s in new.samples}
            shared = a.keys() & b.keys()
            if shared:
                scores[i, j] = sum(box_iou(a[k], b[k]) for k in shared) / len(shared)
    for i, j in zip(*linear_sum_assignment(-scores)):
        row, col = sorted(scores[i], reverse=True), sorted(scores[:, j], reverse=True)
        unique = (len(row) < 2 or row[0] - row[1] >= 0.05) and (
            len(col) < 2 or col[0] - col[1] >= 0.05
        )
        if scores[i, j] >= 0.2 and unique:
            new_tracks[j] = new_tracks[j].model_copy(update={"actor_id": old_tracks[i].actor_id})
    return new_tracks


def spatial_stable(old_tracks, new_tracks):
    if not old_tracks or len(old_tracks) != len(new_tracks):
        return False
    new_by_id = {t.actor_id: t for t in new_tracks}
    for old in old_tracks:
        new = new_by_id.get(old.actor_id)
        if new is None or "ambiguous" in {old.association_status, new.association_status}:
            return False
        a = {s.frame_id: s.bbox for s in old.samples}
        b = {s.frame_id: s.bbox for s in new.samples}
        shared = a.keys() & b.keys()
        if not shared or sum(box_iou(a[k], b[k]) for k in shared) / len(shared) < 0.9:
            return False
    return True


async def refine(candidate, tracks, sample_frames, infer, localizer, settings, budget):
    current, current_tracks = candidate, tracks
    revisions: list[dict] = []
    seen: set[tuple] = set()
    evidence: set[str] = set(candidate.evidence_frame_ids)
    stable = 0
    seen_intervals = {(current.start_sec, current.end_sec)}
    for iteration in range(min(settings.max_rounds, settings.boundary_calls)):
        signature = (current.start_sec, current.end_sec, tuple(sorted(evidence)))
        if signature in seen:
            return current, current_tracks, revisions, "repeated_state"
        seen.add(signature)
        try:
            budget.check()
            frames = await sample_frames(current, iteration)
            ids = {f.frame_id for f in frames}
            if not ids - evidence:
                return current, current_tracks, revisions, "no_new_evidence"
            evidence |= ids
            spatial = await localizer.localize(current, frames) if localizer else current_tracks
            spatial = align_tracks(current_tracks, spatial)
            proposed = await infer(
                ModelRequest(
                    task="refine",
                    query=current.label,
                    frames=frames,
                    start_sec=min(f.timestamp_sec for f in frames),
                    end_sec=max(current.end_sec, max(f.timestamp_sec for f in frames) + 1e-6),
                    context={
                        "prediction": current.model_dump(mode="json"),
                        "tracks": [t.model_dump(mode="json") for t in spatial],
                    },
                )
            )
            update = (
                supported_update(current, proposed.predictions[0], proposed.presence, frames)
                if len(proposed.predictions) == 1
                else None
            )
            if update is None:
                revisions.append(
                    {"iteration": iteration, "accepted": False, "reason": "unsupported_update"}
                )
                return current, current_tracks, revisions, "invalid_refinement"
            interval = (update.start_sec, update.end_sec)
            if interval in seen_intervals and interval != (current.start_sec, current.end_sec):
                revisions.append(
                    {"iteration": iteration, "accepted": False, "reason": "oscillation"}
                )
                return current, current_tracks, revisions, "oscillation"
            seen_intervals.add(interval)
            converged = (
                max(
                    abs(update.start_sec - current.start_sec), abs(update.end_sec - current.end_sec)
                )
                <= settings.boundary_target_sec
            )
            if localizer:
                converged = converged and spatial_stable(current_tracks, spatial)
            stable = stable + 1 if converged else 0
            revisions.append(
                {
                    "iteration": iteration,
                    "accepted": True,
                    "before": [current.start_sec, current.end_sec],
                    "after": [update.start_sec, update.end_sec],
                }
            )
            current = update.model_copy(
                update={
                    "actor_ids": [t.actor_id for t in spatial],
                    "spatial_status": "partial"
                    if spatial
                    else ("unlocalized" if localizer else "disabled"),
                }
            )
            current_tracks = spatial
            if stable >= 2:
                return current, current_tracks, revisions, "converged"
        except Exception as exc:
            from shar.errors import Cancelled, SharError

            if isinstance(exc, Cancelled):
                raise
            if not isinstance(exc, SharError):
                raise
            revisions.append({"iteration": iteration, "accepted": False, "reason": exc.code})
            return current, current_tracks, revisions, exc.code.lower()
    return current, current_tracks, revisions, "max_iterations"
