from shar.evaluation.matching import joint_overlap, temporal_matches
from shar.inference.temporal.merge import tiou


def average_precision(hits, total_gt):
    if total_gt == 0:
        return None
    tp, precision, recall = 0, [0.0], [0.0]
    for rank, hit in enumerate(hits, 1):
        tp += int(hit)
        precision.append(tp / rank)
        recall.append(tp / total_gt)
    precision.append(0.0)
    recall.append(1.0)
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    return sum(
        (recall[i] - recall[i - 1]) * precision[i]
        for i in range(1, len(recall))
        if recall[i] != recall[i - 1]
    )


def recognition_metrics(predicted: dict[str, set], truth: dict[str, set]):
    classes = set().union(*truth.values(), *predicted.values())
    tp = fp = fn = 0
    per_class = {}
    for label in sorted(classes):
        a = sum(label in predicted.get(v, set()) and label in labels for v, labels in truth.items())
        b = sum(
            label in predicted.get(v, set()) and label not in labels for v, labels in truth.items()
        )
        c = sum(
            label not in predicted.get(v, set()) and label in labels for v, labels in truth.items()
        )
        tp, fp, fn = tp + a, fp + b, fn + c
        per_class[label] = 2 * a / (2 * a + b + c) if 2 * a + b + c else 0.0
    return {
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "micro_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "macro_f1": sum(per_class.values()) / len(per_class) if per_class else 0.0,
        "exact_set_accuracy": sum(predicted.get(v, set()) == labels for v, labels in truth.items())
        / len(truth)
        if truth
        else 0.0,
    }


def temporal_metrics(predictions, truth, thresholds=(0.5, 0.75, 0.95)):
    classes = sorted({p.label for _, p in truth})
    absent = sorted({p.label for _, p in predictions} - set(classes))
    results = {}
    for threshold in thresholds:
        aps, errors, hits = {}, [], 0
        for label in classes:
            pred = [(v, p) for v, p in predictions if p.label == label]
            gt = [(v, p) for v, p in truth if p.label == label]
            matches = temporal_matches(pred, gt, threshold)
            aps[label] = average_precision([g is not None for _, g, _ in matches], len(gt))
            for p, g, _ in matches:
                if g is not None:
                    hits += 1
                    errors.extend([abs(p.start_sec - g.start_sec), abs(p.end_sec - g.end_sec)])
        results[str(threshold)] = {
            "AP": aps,
            "mAP": sum(aps.values()) / len(aps) if aps else None,
            "boundary_mae_sec": sum(errors) / len(errors) if errors else None,
            "recall": hits / len(truth) if truth else None,
        }
    short = [(v, p) for v, p in truth if p.end_sec - p.start_sec < 2.0]
    missed = sum(
        not any(
            v == pv and p.label == pp.label and tiou(p, pp) >= thresholds[0]
            for pv, pp in predictions
        )
        for v, p in short
    )
    no_action = {v for v, _ in predictions} - {v for v, _ in truth}
    return {
        "thresholds": results,
        "absent_classes": absent,
        "unknown_class_false_positives": sum(p.label in absent for _, p in predictions),
        "action_free_false_positives": sum(v in no_action for v, _ in predictions),
        "missed_short_action_rate": missed / len(short) if short else None,
    }


def joint_metrics(
    reports, annotations, temporal_threshold=0.5, spatial_threshold=0.5, tolerance=0.125
):
    gt_lookup = {a.video_id: a for a in annotations}
    labels = sorted({p.label for a in annotations for p in a.actions if p.actor_ids})
    ap, overlaps, coverages = {}, [], []
    for label in labels:
        gt = [
            (a.video_id, p)
            for a in annotations
            for p in a.actions
            if p.label == label and p.actor_ids
        ]
        pred = [(r.video_id, p, r) for r in reports for p in r.predictions if p.label == label]
        used, hits = set(), []
        for video, p, report in sorted(
            pred, key=lambda item: (-(item[1].confidence or 0.0), item[0], item[1].id)
        ):
            candidates = []
            for i, (gv, g) in enumerate(gt):
                if gv != video or i in used or tiou(p, g) < temporal_threshold:
                    continue
                overlap, coverage = joint_overlap(
                    p,
                    g,
                    {t.actor_id: t for t in report.tracks},
                    {t.actor_id: t for t in gt_lookup[video].tracks},
                    tolerance,
                )
                if overlap is not None:
                    candidates.append((overlap, tiou(p, g), i, coverage))
            best = max(candidates, default=(0.0, 0.0, -1, 0.0))
            hit = best[2] >= 0 and best[0] >= spatial_threshold
            hits.append(hit)
            if hit:
                used.add(best[2])
        ap[label] = average_precision(hits, len(gt))
    # Spatial coverage includes unmatched GT events as zero, not only successful matches.
    report_lookup = {r.video_id: r for r in reports}
    for annotation in annotations:
        report = report_lookup.get(annotation.video_id)
        for gt in annotation.actions:
            if not gt.actor_ids:
                continue
            candidates = (
                [
                    p
                    for p in report.predictions
                    if p.label == gt.label and tiou(p, gt) >= temporal_threshold
                ]
                if report
                else []
            )
            scores = [
                joint_overlap(
                    p,
                    gt,
                    {t.actor_id: t for t in report.tracks},
                    {t.actor_id: t for t in annotation.tracks},
                    tolerance,
                )
                for p in candidates
                if report is not None
            ]
            spatial_best = max((s for s in scores if s[0] is not None), default=(0.0, 0.0))
            overlaps.append(spatial_best[0])
            coverages.append(spatial_best[1])
    return {
        "mAP": sum(ap.values()) / len(ap) if ap else None,
        "AP": ap,
        "mean_visible_iou": sum(overlaps) / len(overlaps) if overlaps else None,
        "localization_coverage": sum(coverages) / len(coverages) if coverages else None,
        "conditional_visible_iou": sum(overlaps) / sum(coverages) if sum(coverages) else None,
        "sampling_tolerance_sec": tolerance,
        "protocol": "project-defined sparse joint metric",
    }
