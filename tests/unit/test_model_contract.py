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


def test_provider_cooldown_survives_restart_and_expires(settings, monkeypatch):
    from shar.inference.multimodal.limits import ProviderCooldown

    clock = [1000.0]
    monkeypatch.setattr("shar.inference.multimodal.limits.time.time", lambda: clock[0])
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    error = SharError("PROVIDER_DAILY_QUOTA", "Daily quota exhausted.", False, 429)
    error.retry_after = 60
    ProviderCooldown(settings).record(error)
    with pytest.raises(SharError) as caught:
        ProviderCooldown(settings).check()
    assert caught.value.retry_after == 60
    assert not caught.value.retryable
    assert "test-key" not in (settings.storage_root / "quotas.sqlite").read_bytes().decode(
        errors="ignore"
    )
    clock[0] += 61
    ProviderCooldown(settings).check()


def test_provider_cooldown_isolated_on_credential_change(settings, monkeypatch):
    from shar.inference.multimodal.limits import ProviderCooldown

    monkeypatch.setenv("GEMINI_API_KEY", "project-a-key")
    error = SharError("PROVIDER_DAILY_QUOTA", "Daily quota exhausted.", False, 429)
    error.retry_after = 60
    ProviderCooldown(settings).record(error)
    monkeypatch.setenv("GEMINI_API_KEY", "project-b-key")
    ProviderCooldown(settings).check()
    monkeypatch.setenv("GEMINI_API_KEY", "project-a-key")
    with pytest.raises(SharError):
        ProviderCooldown(settings).check()
    ProviderCooldown(settings).clear()
    ProviderCooldown(settings).check()


def test_local_throttle_reports_when_capacity_returns(tmp_path, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr("shar.inference.multimodal.limits.time.time", lambda: clock[0])
    limiter = SharedLimiter(tmp_path, "model", 1, 100, 10)
    limiter.reserve(50)
    clock[0] += 10
    with pytest.raises(SharError) as caught:
        limiter.reserve(30)
    assert caught.value.retry_after == 50
    assert caught.value.retryable
    clock[0] += 51
    limiter.reserve(30)
    with pytest.raises(SharError) as caught:
        limiter.reserve(101)
    assert caught.value.code == "REQUEST_BUDGET" and not caught.value.retryable


def test_daily_quota_reports_rolling_recovery_without_consuming_calls(tmp_path, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr("shar.inference.multimodal.limits.time.time", lambda: clock[0])
    limiter = SharedLimiter(tmp_path, "model", 10, 1000, 2)
    limiter.reserve(20)
    clock[0] += 100
    limiter.reserve(20)
    with pytest.raises(SharError) as caught:
        limiter.reserve(20)
    assert caught.value.code == "DAILY_QUOTA"
    assert caught.value.status == 429
    assert not caught.value.retryable  # Approval is required; never sleep/retry automatically.
    assert caught.value.retry_after == 86300
    assert "rolling 24-hour" in caught.value.message
    # Lowering a cap requires enough reservations to expire, not just the oldest.
    lowered = SharedLimiter(tmp_path, "model", 10, 1000, 1)
    assert lowered.daily_retry_after() == 86400
    clock[0] = 87400
    assert limiter.daily_retry_after() == 0
    limiter.reserve(20)
    assert limiter.daily_retry_after() == 100
