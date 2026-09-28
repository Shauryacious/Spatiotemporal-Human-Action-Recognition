import hashlib
import json
from pathlib import Path

from pydantic import Field

from shar.schemas import ActionPrediction, ActorTrack, Contract


class DatasetVideo(Contract):
    video_id: str
    path: str
    sha256: str
    duration_sec: float = Field(gt=0)
    split: str
    group_id: str
    license: str = Field(min_length=1)
    source_id: str | None = None
    subject_id: str | None = None


class Annotation(Contract):
    video_id: str
    actions: list[ActionPrediction] = Field(default_factory=list)
    tracks: list[ActorTrack] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)


def read_jsonl(path: Path, model):
    return [
        model.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()
    ]


def validate_splits(records):
    seen: dict[tuple[str, str], str] = {}
    identifiers: set[str] = set()
    for record in records:
        if record.video_id in identifiers:
            raise ValueError("Duplicate video ID")
        identifiers.add(record.video_id)
        for kind in ("sha256", "group_id", "source_id", "subject_id"):
            value = getattr(record, kind)
            if value is None:
                continue
            key = (kind, value)
            if key in seen and seen[key] != record.split:
                raise ValueError(f"Split leakage via {kind}")
            seen[key] = record.split


def load_dataset(manifest: Path, annotations: Path, verify_media=True):
    records = read_jsonl(manifest, DatasetVideo)
    validate_splits(records)
    ground_truth = read_jsonl(annotations, Annotation)
    lookup = {v.video_id: v for v in records}
    if len({a.video_id for a in ground_truth}) != len(ground_truth):
        raise ValueError("Duplicate annotation record")
    for annotation in ground_truth:
        if annotation.video_id not in lookup:
            raise ValueError("Annotation references unknown video")
        actors = {t.actor_id for t in annotation.tracks}
        for action in annotation.actions:
            if action.end_sec > lookup[annotation.video_id].duration_sec:
                raise ValueError("Ground truth exceeds video duration")
            if not set(action.actor_ids) <= actors:
                raise ValueError("Unknown ground truth actor")
    if {a.video_id for a in ground_truth} != set(lookup):
        raise ValueError("Every video requires an annotation record, including empty scenes")
    if verify_media:
        for record in records:
            path = manifest.parent / record.path
            with path.open("rb") as source:
                digest = hashlib.file_digest(source, "sha256").hexdigest()
            if digest != record.sha256:
                raise ValueError("Video hash mismatch")
    return records, ground_truth


class EvaluationSettings(Contract):
    temporal_thresholds: list[float] = Field(
        default_factory=lambda: [0.5, 0.75, 0.95], min_length=1
    )
    spatial_threshold: float = Field(default=0.5, gt=0, le=1)
    sampling_tolerance_sec: float = Field(default=0.125, ge=0)
    unknown_score_rank: float = Field(default=0.0, ge=0, le=1)
    aliases: dict[str, str] = Field(default_factory=dict)
    seed: int = 0

    @classmethod
    def load(cls, path: Path | None = None):
        import yaml

        value = yaml.safe_load(path.read_text()) if path else {}
        result = cls.model_validate_json(json.dumps(value or {}))
        if any(not 0 < t <= 1 for t in result.temporal_thresholds):
            raise ValueError("Temporal evaluation thresholds must be in (0,1]")
        return result
