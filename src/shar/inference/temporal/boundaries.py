from shar.schemas import BoundaryEstimate


def brackets(prediction, presence, frames):
    """Local absent/present transitions; no global monotonicity assumption."""
    states = {p.frame_id: p.state for p in presence}
    observations = sorted((f.timestamp_sec, states.get(f.frame_id, "uncertain")) for f in frames)
    starts, ends = [], []
    for (left, a), (right, b) in zip(observations, observations[1:]):
        if a == "absent" and b == "present":
            starts.append((left, right))
        if a == "present" and b == "absent":
            ends.append((left, right))

    def bound(value, candidates):
        supported = [(lo, hi) for lo, hi in candidates if lo <= value <= hi]
        if not supported:
            return None
        lo, hi = min(supported, key=lambda p: p[1] - p[0])
        return BoundaryEstimate(estimate=value, lower=lo, upper=hi)

    return bound(prediction.start_sec, starts), bound(prediction.end_sec, ends)


def supported_update(old, proposed, presence, frames):
    start, end = brackets(proposed, presence, frames)
    if abs(old.start_sec - proposed.start_sec) > 1e-6 and start is None:
        return None
    if abs(old.end_sec - proposed.end_sec) > 1e-6 and end is None:
        return None
    if old.label != proposed.label:
        return None
    return proposed.model_copy(
        update={
            "id": old.id,
            "start_boundary": start or old.start_boundary,
            "end_boundary": end or old.end_boundary,
        }
    )
