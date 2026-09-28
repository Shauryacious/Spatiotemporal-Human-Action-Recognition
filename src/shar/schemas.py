from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


def uid() -> str:
    return uuid4().hex


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)


class VideoMetadata(Contract):
    duration_sec: float = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    source_width: int = Field(gt=0)
    source_height: int = Field(gt=0)
    rotation: int = 0
    time_base: str
    first_pts: int
    nominal_fps: float | None = None
    frame_count: int | None = None
    count_reliability: str = "reported"
    codec: str = "unknown"


class VideoAsset(Contract):
    id: str = Field(default_factory=uid)
    owner_id: str
    sha256: str
    managed_uri: str
    bytes: int = Field(gt=0)
    format: str
    metadata: VideoMetadata


class Transform(Contract):
    source_width: int
    source_height: int
    prepared_width: int
    prepared_height: int
    crop_x: float = 0.0
    crop_y: float = 0.0
    crop_width: float = 1.0
    crop_height: float = 1.0


class FrameRecord(Contract):
    frame_id: str
    frame_index: int = Field(ge=0)
    pts: int
    timestamp_sec: float = Field(ge=0)
    transform: Transform
    image_hash: str
    jpeg: bytes = Field(exclude=True, repr=False)


class BoundaryEstimate(Contract):
    estimate: float = Field(ge=0)
    lower: float = Field(ge=0)
    upper: float = Field(ge=0)
    truncated: bool = False

    @model_validator(mode="after")
    def ordered(self):
        if not self.lower <= self.estimate <= self.upper:
            raise ValueError("boundary bounds must contain estimate")
        return self


class BBox(Contract):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    width: float = Field(gt=0, le=1)
    height: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def within(self):
        if self.x + self.width > 1 or self.y + self.height > 1:
            raise ValueError("box exceeds displayed image")
        return self


class SpatialDetection(Contract):
    frame_id: str
    frame_index: int = Field(ge=0)
    timestamp_sec: float = Field(ge=0)
    bbox: BBox
    confidence: float | None = Field(default=None, ge=0, le=1)
    source: str = "mllm"
    observed: bool = True


class ActorTrack(Contract):
    actor_id: str = Field(default_factory=uid)
    association_status: Literal["provisional", "ambiguous", "resolved"] = "provisional"
    samples: list[SpatialDetection] = Field(default_factory=list, max_length=10000)

    @model_validator(mode="after")
    def ordered(self):
        times = [s.timestamp_sec for s in self.samples]
        if times != sorted(set(times)):
            raise ValueError("track observations must have unique increasing times")
        return self


class ActionPrediction(Contract):
    id: str = Field(default_factory=uid)
    label: str = Field(min_length=1, max_length=300, pattern=r"\S")
    start_sec: float = Field(ge=0)
    end_sec: float = Field(gt=0)
    confidence: float | None = Field(default=None, ge=0, le=1)
    confidence_kind: str = "model_self_reported_uncalibrated"
    start_boundary: BoundaryEstimate | None = None
    end_boundary: BoundaryEstimate | None = None
    actor_ids: list[str] = Field(default_factory=list, max_length=100)
    evidence_frame_ids: list[str] = Field(default_factory=list, max_length=1024)
    description: str = Field(default="", max_length=2000)
    spatial_status: Literal["disabled", "unlocalized", "partial", "localized"] = "disabled"
    continues_before: bool = False
    continues_after: bool = False

    @model_validator(mode="after")
    def interval(self):
        if self.start_sec >= self.end_sec:
            raise ValueError("interval must have positive duration")
        for bound, value in (
            (self.start_boundary, self.start_sec),
            (self.end_boundary, self.end_sec),
        ):
            if bound and abs(bound.estimate - value) > 1e-6:
                raise ValueError("boundary estimate differs from interval")
        return self


class Recognition(Contract):
    label: str = Field(min_length=1, max_length=300, pattern=r"\S")
    confidence: float | None = Field(default=None, ge=0, le=1)
    evidence_frame_ids: list[str] = Field(default_factory=list, max_length=1024)


class Presence(Contract):
    frame_id: str
    state: Literal["present", "absent", "uncertain"]


class ModelPayload(Contract):
    predictions: list[ActionPrediction] = Field(default_factory=list, max_length=100)
    recognition: list[Recognition] = Field(default_factory=list, max_length=100)
    tracks: list[ActorTrack] = Field(default_factory=list, max_length=100)
    presence: list[Presence] = Field(default_factory=list, max_length=1024)


class Coverage(Contract):
    analyzed_intervals: list[tuple[float, float]] = Field(default_factory=list)
    failed_intervals: list[tuple[float, float]] = Field(default_factory=list)


class Usage(Contract):
    model_calls: int = 0
    cache_hits: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    estimated_cost: float | None = None


class ActionReport(Contract):
    schema_version: str = "1.0"
    job_id: str = Field(default_factory=uid)
    video_id: str
    status: Literal["succeeded", "partial"] = "succeeded"
    video: VideoMetadata
    predictions: list[ActionPrediction] = Field(default_factory=list)
    recognition: list[Recognition] = Field(default_factory=list)
    tracks: list[ActorTrack] = Field(default_factory=list)
    coverage: Coverage = Field(default_factory=Coverage)
    warnings: list[str] = Field(default_factory=list)
    provenance: dict = Field(default_factory=dict)
    usage: Usage = Field(default_factory=Usage)
    revisions: list[dict] = Field(default_factory=list)

    @model_validator(mode="after")
    def references(self):
        actors = {t.actor_id for t in self.tracks}
        for p in self.predictions:
            if p.end_sec > self.video.duration_sec + 1e-6:
                raise ValueError("prediction exceeds video")
            if not set(p.actor_ids) <= actors:
                raise ValueError("prediction references unknown actor")
        for start, end in self.coverage.analyzed_intervals + self.coverage.failed_intervals:
            if not 0 <= start < end <= self.video.duration_sec + 1e-6:
                raise ValueError("invalid coverage interval")
        return self


class JobStatus(Contract):
    id: str
    status: Literal[
        "queued", "running", "retry_wait", "succeeded", "partial", "failed", "cancelled"
    ]
    stage: str
    created_at: float
    started_at: float | None = None
    completed_at: float | None = None
    attempt: int = Field(ge=0)
    cancel_requested: bool
    error_code: str | None = None
    calls: int = Field(ge=0)
    next_attempt_at: float
