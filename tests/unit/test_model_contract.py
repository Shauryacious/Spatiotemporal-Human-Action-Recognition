import time

import pytest
from conftest import frame

from shar.errors import SharError
from shar.inference.multimodal.base import ModelRequest
from shar.inference.multimodal.cache import InferenceCache
from shar.inference.multimodal.limits import Budget, SharedLimiter
from shar.inference.multimodal.mock import MockModel


async def test_mock():
    model = MockModel(demo=True)
    response = await model.infer(ModelRequest("recognize", "walk", [frame()], 0.0, 1.0))
    assert response.payload.recognition[0].label == "synthetic demonstration"
    assert not response.payload.predictions


def test_limits(tmp_path):
    budget = Budget(1, time.monotonic() + 10)
    budget.reserve()
    with pytest.raises(SharError):
        budget.reserve()
    a = SharedLimiter(tmp_path, "model", 1, 100, 2)
    b = SharedLimiter(tmp_path, "model", 1, 100, 2)
    a.reserve(20)
    with pytest.raises(SharError):
        b.reserve(20)


def test_cache_scope(settings):
    request = ModelRequest("temporal", "walk", [frame()], 0.0, 1.0)
    assert InferenceCache.key("a", "video", request, settings) != InferenceCache.key(
        "b", "video", request, settings
    )
    old = InferenceCache.key("a", "video", request, settings)
    request.frames = [frame(1, 0.5)]
    assert old != InferenceCache.key("a", "video", request, settings)


def test_cache_preserves_resolved_model(settings):
    cache = InferenceCache(settings.storage_root, 86400)
    cache.put("key", {"predictions": []}, "resolved-v2")
    assert cache.get("key") == {"payload": {"predictions": []}, "model": "resolved-v2"}
