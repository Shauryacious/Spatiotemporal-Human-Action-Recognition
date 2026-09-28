import numpy as np
from scipy.optimize import linear_sum_assignment

from shar.inference.spatial.association import box_iou
from shar.inference.temporal.merge import tiou


def temporal_matches(predictions, truth, threshold, score=lambda p: p.confidence or 0.0):
    used, matches = set(), []
    for video, pred in sorted(predictions, key=lambda item: (-score(item[1]), item[0], item[1].id)):
        options = [
            (tiou(pred, gt), index)
            for index, (gt_video, gt) in enumerate(truth)
            if gt_video == video and gt.label == pred.label and index not in used
        ]
        overlap, index = max(options, default=(0.0, -1))
        matched = index >= 0 and overlap >= threshold
        if matched:
            used.add(index)
        matches.append((pred, truth[index][1] if matched else None, overlap if matched else 0.0))
    return matches


def track_overlap(pred_track, gt_track, prediction, tolerance):
    gt = [s for s in gt_track.samples if s.observed]
    available = list(pred_track.samples) if pred_track else []
    used, overlaps, observed = set(), [], 0
    for sample in gt:
        options = [
            (abs(p.timestamp_sec - sample.timestamp_sec), i, p)
            for i, p in enumerate(available)
            if i not in used
            and p.observed
            and abs(p.timestamp_sec - sample.timestamp_sec) <= tolerance
            and prediction.start_sec <= sample.timestamp_sec < prediction.end_sec
        ]
        if not options:
            overlaps.append(0.0)
        else:
            _, i, selected = min(options, key=lambda item: (item[0], item[1]))
            used.add(i)
            observed += 1
            overlaps.append(box_iou(selected.bbox, sample.bbox))
    return (sum(overlaps) / len(gt) if gt else 0.0, observed / len(gt) if gt else 0.0)


def joint_overlap(prediction, gt, pred_tracks, gt_tracks, tolerance=0.125):
    targets = [gt_tracks[a] for a in gt.actor_ids if a in gt_tracks]
    candidates = [
        pred_tracks[a]
        for a in prediction.actor_ids
        if a in pred_tracks and pred_tracks[a].association_status != "ambiguous"
    ]
    if not targets:
        return None, None
    scores = np.zeros((len(targets), max(len(targets), len(candidates))))
    coverage = np.zeros_like(scores)
    for i, target in enumerate(targets):
        # Only annotated visible observations in this action count.
        target = target.model_copy(
            update={
                "samples": [
                    s for s in target.samples if gt.start_sec <= s.timestamp_sec < gt.end_sec
                ]
            }
        )
        for j, candidate in enumerate(candidates):
            scores[i, j], coverage[i, j] = track_overlap(candidate, target, prediction, tolerance)
    rows, columns = linear_sum_assignment(-scores)
    return float(scores[rows, columns].sum() / len(targets)), float(
        coverage[rows, columns].sum() / len(targets)
    )
