import hashlib
import json
import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    provider: Literal["mock", "gemini"] = "mock"
    model: str = "fixture-v1"
    input_strategy: Literal["frames", "native_video"] = "frames"
    coarse_fps: float = Field(default=1, gt=0, le=60)
    fine_fps: float = Field(default=4, gt=0, le=60)
    window_sec: float = Field(default=30, gt=0)
    overlap_sec: float = Field(default=5, ge=0)
    longest_edge: int = Field(default=640, ge=64, le=4096)
    jpeg_quality: int = Field(default=85, ge=1, le=100)
    grayscale: bool = False
    motion_sampling: bool = False
    motion_extra_frames: int = Field(default=8, ge=0, le=64)
    motion_threshold: float = Field(default=0.08, ge=0, le=1)
    max_frames: int = Field(default=64, ge=2, le=1024)
    max_request_bytes: int = Field(default=18_000_000, gt=0)
    max_candidates: int = Field(default=100, gt=0, le=1000)
    max_calls: int = Field(default=100, gt=0)
    max_rounds: int = Field(default=3, ge=0, le=20)
    boundary_calls: int = Field(default=8, ge=0)
    boundary_target_sec: float = Field(default=0.25, gt=0)
    proposal_threshold: float = Field(default=0.3, ge=0, le=1)
    nms_tiou: float = Field(default=0.7, gt=0, le=1)
    enable_stitching: bool = True
    enable_temporal: bool = True
    enable_spatial: bool = False
    enable_refinement: bool = True
    enable_boundary_search: bool = True
    call_timeout_sec: float = Field(default=60, gt=0)
    provider_max_attempts: int = Field(default=3, ge=1, le=10)
    require_retry_approval: bool = True
    retry_base_sec: float = Field(default=2, gt=0)
    retry_max_sec: float = Field(default=30, gt=0)
    upload_timeout_sec: float = Field(default=120, gt=0)
    job_timeout_sec: float = Field(default=1800, gt=0)
    max_upload_bytes: int = Field(default=1073741824, gt=0)
    max_duration_sec: float = Field(default=3600, gt=0)
    max_pixels: int = Field(default=33_177_600, gt=0)
    output_tokens: int = Field(default=4096, gt=0)
    thinking_level: Literal["minimal", "low", "medium", "high"] | None = None
    estimated_tokens_per_frame: int = Field(default=512, gt=0)
    rpm: int | None = Field(default=None, gt=0)
    tpm: int | None = Field(default=None, gt=0)
    rpd: int | None = Field(default=None, gt=0)
    input_price_per_million: float | None = Field(default=None, ge=0)
    output_price_per_million: float | None = Field(default=None, ge=0)
    max_cost: float | None = Field(default=None, gt=0)
    storage_root: Path = Path("runtime")
    host: str = "127.0.0.1"
    auth_mode: Literal["local", "token"] = "local"
    retention_hours: float = Field(default=24, gt=0)
    lease_sec: float = Field(default=90, ge=3)
    max_job_attempts: int = Field(default=3, ge=1)
    max_queued_jobs: int = Field(default=100, ge=1)
    stream_window_sec: float = Field(default=8, gt=0)
    stream_stride_sec: float = Field(default=4, gt=0)
    stream_ring_sec: float = Field(default=16, gt=0)
    stream_count: int = Field(default=1, ge=1)
    stream_sources: dict[str, str | int] = Field(default_factory=dict)
    cache_ttl_sec: float = Field(default=86400, ge=0)
    seed: int = 0

    @model_validator(mode="after")
    def consistent(self):
        if self.retry_max_sec < self.retry_base_sec:
            raise ValueError("retry_max_sec must be at least retry_base_sec")
        if self.overlap_sec >= self.window_sec:
            raise ValueError("overlap_sec must be smaller than window_sec")
        if not self.enable_temporal and (self.enable_spatial or self.enable_refinement):
            raise ValueError("spatial/refinement requires temporal localization")
        if self.auth_mode == "local" and self.host not in {"127.0.0.1", "::1", "localhost"}:
            raise ValueError("local authentication requires loopback binding")
        if self.provider == "gemini" and (
            self.model == "fixture-v1" or not self.rpm or not self.tpm or not self.rpd
        ):
            raise ValueError("Gemini requires an explicit model and account RPM/TPM/RPD")
        if self.stream_ring_sec < self.stream_window_sec:
            raise ValueError("stream ring must contain a complete window")
        if self.max_cost is not None and (
            self.input_price_per_million is None or self.output_price_per_million is None
        ):
            raise ValueError("spend budgets require recorded input/output prices")
        return self

    def public(self) -> dict:
        value = self.model_dump(mode="json")
        value.pop("stream_sources")
        value.pop("storage_root")
        return value

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.public(), sort_keys=True).encode()).hexdigest()


def load_settings(path: Path | None = None, overrides: dict | None = None) -> Settings:
    values = yaml.safe_load(path.read_text()) or {} if path else {}
    for key in (
        "provider",
        "model",
        "storage_root",
        "host",
        "auth_mode",
        "rpm",
        "tpm",
        "rpd",
        "max_calls",
        "max_job_attempts",
        "output_tokens",
        "thinking_level",
        "call_timeout_sec",
        "provider_max_attempts",
        "require_retry_approval",
        "retry_base_sec",
        "retry_max_sec",
    ):
        if os.environ.get(f"SHAR_{key.upper()}"):
            values[key] = os.environ[f"SHAR_{key.upper()}"]
    if overrides:
        values.update(overrides)
    return Settings.model_validate(values)
