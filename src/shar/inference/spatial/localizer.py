from typing import Protocol

from shar.inference.multimodal.base import ModelRequest
from shar.inference.spatial.association import associate


class SpatialLocalizer(Protocol):
    async def localize(self, prediction, frames): ...


class MLLMLocalizer:
    def __init__(self, infer, query):
        self.infer, self.query = infer, query

    async def localize(self, prediction, frames):
        inside = [f for f in frames if prediction.start_sec <= f.timestamp_sec < prediction.end_sec]
        if not inside:
            return []
        response = await self.infer(
            ModelRequest(
                task="spatial",
                query=self.query,
                frames=inside,
                start_sec=prediction.start_sec,
                end_sec=prediction.end_sec,
                context={"prediction": prediction.model_dump(mode="json")},
            )
        )
        observations = [d for track in response.tracks for d in track.samples]
        # Provider IDs are hypotheses. Re-associate locally with explicit gates.
        return associate(observations)
