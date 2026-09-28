from shar.schemas import ActionPrediction


def tiou(a, b) -> float:
    start_a, end_a = (a.start_sec, a.end_sec) if hasattr(a, "start_sec") else a
    start_b, end_b = (b.start_sec, b.end_sec) if hasattr(b, "start_sec") else b
    intersection = max(0.0, min(end_a, end_b) - max(start_a, start_b))
    union = end_a - start_a + end_b - start_b - intersection
    return intersection / union if union > 0 else 0.0


def same_actor(a: ActionPrediction, b: ActionPrediction) -> bool:
    # Unknown identities are not evidence of sameness. Exact evidence can identify
    # duplicate temporal proposals before spatial association is available.
    return (bool(a.actor_ids) and set(a.actor_ids) == set(b.actor_ids)) or (
        not a.actor_ids
        and not b.actor_ids
        and bool(set(a.evidence_frame_ids) & set(b.evidence_frame_ids))
    )


def nms(predictions, threshold=0.7):
    kept: list[ActionPrediction] = []
    for candidate in sorted(
        predictions, key=lambda p: (-(p.confidence if p.confidence is not None else -1), p.id)
    ):
        if not any(
            candidate.label == other.label
            and same_actor(candidate, other)
            and tiou(candidate, other) >= threshold
            for other in kept
        ):
            kept.append(candidate)
    return sorted(kept, key=lambda p: (p.start_sec, p.label, p.id))


def stitch(predictions):
    result: list[ActionPrediction] = []
    for candidate in sorted(predictions, key=lambda p: (p.start_sec, p.end_sec)):
        for i, previous in enumerate(result):
            if (
                previous.label == candidate.label
                and same_actor(previous, candidate)
                and previous.continues_after
                and candidate.continues_before
                and candidate.start_sec < previous.end_sec
                and set(previous.evidence_frame_ids) & set(candidate.evidence_frame_ids)
            ):
                data = previous.model_dump()
                if candidate.end_sec > previous.end_sec:
                    data.update(
                        end_sec=candidate.end_sec,
                        end_boundary=candidate.end_boundary,
                        continues_after=candidate.continues_after,
                    )
                data["evidence_frame_ids"] = sorted(
                    set(previous.evidence_frame_ids + candidate.evidence_frame_ids)
                )
                result[i] = ActionPrediction.model_validate(data)
                break
        else:
            result.append(candidate)
    return result
