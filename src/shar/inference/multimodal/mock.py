import asyncio
from collections import deque

from shar.inference.multimodal.base import ModelCapabilities, ModelRequest, ModelResponse
from shar.schemas import ActionPrediction, ModelPayload, Recognition, Usage


class MockModel:
    """Deterministic scripted adapter. Default empty output makes no semantic claims."""

    def __init__(self, responses=None, delay: float = 0.0, demo: bool = False):
        self.responses = deque(responses or [])
        self.delay, self.demo, self.calls = delay, demo, 0

    def capabilities(self):
        return ModelCapabilities()

    def estimate_usage(self, request):
        return len(request.frames) * 512 + len(request.prompt) // 4

    async def infer(self, request: ModelRequest) -> ModelResponse:
        self.calls += 1
        await asyncio.sleep(self.delay)
        if self.responses:
            value = self.responses.popleft()
            if isinstance(value, Exception):
                raise value
            if callable(value):
                value = value(request)
            if isinstance(value, ModelResponse):
                return value
            return ModelResponse(value, "fixture-v1")
        payload = ModelPayload()
        if self.demo and request.frames:
            evidence = [f.frame_id for f in request.frames]
            if request.task == "recognize":
                payload.recognition = [
                    Recognition(
                        label="synthetic demonstration", confidence=0.8, evidence_frame_ids=evidence
                    )
                ]
            elif request.task == "temporal":
                payload.predictions = [
                    ActionPrediction(
                        label="synthetic demonstration",
                        start_sec=request.start_sec,
                        end_sec=request.end_sec,
                        confidence=0.8,
                        evidence_frame_ids=evidence,
                    )
                ]
        return ModelResponse(
            payload, "fixture-demo-v1" if self.demo else "fixture-v1", Usage(model_calls=1)
        )

    async def close(self):
        pass
