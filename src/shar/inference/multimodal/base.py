from dataclasses import dataclass, field
from typing import Literal, Protocol

from shar.schemas import FrameRecord, ModelPayload, Usage, uid


@dataclass(frozen=True)
class ModelCapabilities:
    images: bool = True
    native_video: bool = False
    structured_output: bool = True
    max_frames: int = 64
    cancellation: bool = True


@dataclass
class ModelRequest:
    task: Literal["recognize", "temporal", "spatial", "refine"]
    query: str
    frames: list[FrameRecord]
    start_sec: float
    end_sec: float
    prompt: str = ""
    context: dict = field(default_factory=dict)
    request_id: str = field(default_factory=uid)
    video_path: str | None = None
    clip_offset_sec: float = 0.0


@dataclass
class ModelResponse:
    payload: str | dict | ModelPayload
    model: str
    usage: Usage = field(default_factory=Usage)
    latency_sec: float = 0.0
    finish_reason: str = "STOP"
    provider_request_id: str | None = None


class MultimodalModel(Protocol):
    def capabilities(self) -> ModelCapabilities: ...
    def estimate_usage(self, request: ModelRequest) -> int: ...
    async def infer(self, request: ModelRequest) -> ModelResponse: ...
    async def close(self) -> None: ...
