import hashlib
import json
from pathlib import Path

import numpy as np

from shar.evaluation.metrics import joint_metrics, recognition_metrics, temporal_metrics
from shar.schemas import ActionReport


def evaluate(
    reports,
    annotations,
    aliases=None,
    thresholds=(0.5, 0.75, 0.95),
    recognition_only=False,
    spatial_threshold=0.5,
    tolerance=0.125,
    unknown_score_rank=0.0,
):
    aliases = aliases or {}
    gt_ids = {a.video_id for a in annotations}
    if len({r.video_id for r in reports}) != len(reports) or any(
        r.video_id not in gt_ids for r in reports
    ):
        raise ValueError("Duplicate or unknown result video")
    mapped = [r.model_copy(deep=True) for r in reports]
    for report in mapped:
        for prediction in report.predictions:
            prediction.label = aliases.get(prediction.label, prediction.label)
            if prediction.confidence is None:
                prediction.confidence = unknown_score_rank
        for label in report.recognition:
            label.label = aliases.get(label.label, label.label)
    recognition = recognition_metrics(
        {
            r.video_id: {p.label for p in r.recognition} | {p.label for p in r.predictions}
            for r in mapped
        },
        {a.video_id: set(a.labels) | {p.label for p in a.actions} for a in annotations},
    )
    result = {
        "recognition": recognition,
        "missing_videos": sorted(gt_ids - {r.video_id for r in reports}),
        "partial_videos": [r.video_id for r in reports if r.status == "partial"],
        "temporal": None,
        "joint": None,
        "efficiency": {
            "model_calls": sum(r.usage.model_calls for r in reports),
            "cache_hits": sum(r.usage.cache_hits for r in reports),
            "calls_per_video_minute": sum(r.usage.model_calls for r in reports)
            / (sum(r.video.duration_sec for r in reports) / 60)
            if reports
            else None,
            "latency_p50_sec": float(
                np.percentile([r.provenance.get("elapsed_sec", 0) for r in reports], 50)
            )
            if reports
            else None,
            "latency_p95_sec": float(
                np.percentile([r.provenance.get("elapsed_sec", 0) for r in reports], 95)
            )
            if reports
            else None,
            "estimated_cost": sum(r.usage.estimated_cost for r in reports)
            if reports and all(r.usage.estimated_cost is not None for r in reports)
            else None,
        },
    }
    if not recognition_only:
        result["temporal"] = temporal_metrics(
            [(r.video_id, p) for r in mapped for p in r.predictions],
            [(a.video_id, p) for a in annotations for p in a.actions],
            thresholds,
        )
        result["joint"] = joint_metrics(
            mapped,
            annotations,
            temporal_threshold=thresholds[0],
            spatial_threshold=spatial_threshold,
            tolerance=tolerance,
        )
    return result


def read_reports(directory: Path):
    return [
        ActionReport.model_validate_json(path.read_text())
        for path in sorted(directory.glob("*.json"))
    ]


def freeze(output: Path, settings, manifest: Path, annotations: Path, extra=None):
    output.mkdir(parents=True, exist_ok=True)
    record = {
        "config": settings.public(),
        "config_hash": settings.fingerprint(),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "annotations_sha256": hashlib.sha256(annotations.read_bytes()).hexdigest(),
        "seed": settings.seed,
        **(extra or {}),
    }
    (output / "manifest.json").write_text(json.dumps(record, indent=2))
    return record
