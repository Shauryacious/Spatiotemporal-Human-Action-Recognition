import asyncio
from unittest.mock import Mock

import httpx
import pytest
from conftest import frame

from shar.errors import Cancelled, SharError
from shar.inference.multimodal.base import ModelRequest
from shar.inference.multimodal.gemini import provider_error
from shar.inference.multimodal.limits import ProviderCooldown
from shar.inference.multimodal.mock import MockModel
from shar.inference.pipeline import Pipeline
from shar.storage.repository import Repository
from shar.workers.runner import Worker


@pytest.mark.parametrize("require_approval", [True, False])
async def test_real_sdk_obeys_retry_approval_policy(settings, asset, require_approval):
    from google import genai
    from google.genai import types

    from shar.inference.multimodal.gemini import GeminiModel

    settings = settings.model_copy(
        update={"retry_base_sec": 0.01, "max_calls": 3, "require_retry_approval": require_approval}
    )
    repo = Repository(settings)
    repo.add_video(asset)
    job = repo.submit(
        "local", {"video_id": asset.id, "prompt": "walk"}, settings.model_dump(mode="json")
    )
    calls = []

    def handle(request):
        calls.append(request)
        if len(calls) < 3:
            return httpx.Response(503, json={"error": {"code": 503, "message": "overloaded"}})
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {"content": {"parts": [{"text": '{"predictions":[]}'}]}, "finishReason": "STOP"}
                ],
                "modelVersion": "test-model",
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        client = genai.Client(
            api_key="fake-key",
            http_options=types.HttpOptions(
                httpx_async_client=http, retry_options=types.HttpRetryOptions(attempts=1)
            ),
        )
        worker = Worker(settings, model_factory=lambda config: GeminiModel(config, client))
        assert await worker.once()
    row = repo.job(job)
    assert row["status"] == ("failed" if require_approval else "succeeded")
    assert row["attempt"] == 1
    assert row["calls"] == len(calls) == (1 if require_approval else 3)
    waits = [e for e in repo.job_activity(job, "local") if e["details"].get("retry_after_sec")]
    assert len(waits) == (0 if require_approval else 2)
    if not require_approval:
        assert waits[1]["details"]["retry_after_sec"] > waits[0]["details"]["retry_after_sec"]
    assert not await worker.once()


async def test_capacity_waits_do_not_consume_provider_attempts(settings, monkeypatch):
    model = MockModel()
    pipeline = Pipeline(settings.model_copy(update={"max_calls": 1}), model=model)
    throttle = SharError("RATE_LIMIT", "wait", True, 429)
    throttle.retry_after = 0.01
    pipeline.limiter = Mock()
    pipeline.limiter.reserve.side_effect = [throttle] * 4 + [1]
    waits = []

    async def wait(task, error, delay):
        waits.append(delay)

    monkeypatch.setattr(pipeline, "wait_to_retry", wait)
    await pipeline.infer(ModelRequest("temporal", "walk", [frame()], 0, 1))
    assert len(waits) == 4
    assert model.calls == pipeline.budget.calls == 1


async def test_shared_transient_cooldown_waits_then_recovers(settings):
    model = MockModel()
    pipeline = Pipeline(settings, model=model)
    pipeline.cooldown = ProviderCooldown(settings)
    error = SharError("PROVIDER_TRANSIENT", "temporarily busy", True, 503)
    error.retry_after = 0.05
    pipeline.cooldown.record(error)
    with pytest.raises(SharError) as caught:
        ProviderCooldown(settings).check()
    assert caught.value.retryable
    await pipeline.infer(ModelRequest("temporal", "walk", [frame()], 0, 1))
    assert model.calls == pipeline.budget.calls == 1


async def test_cancel_while_waiting_makes_no_provider_call(settings):
    cancelled = False
    model = MockModel()
    pipeline = Pipeline(settings, model=model, cancelled=lambda: cancelled)
    pipeline.cooldown = ProviderCooldown(settings)
    error = SharError("PROVIDER_RATE_LIMIT", "wait", True, 429)
    error.retry_after = 60
    pipeline.cooldown.record(error)
    task = asyncio.create_task(pipeline.infer(ModelRequest("temporal", "walk", [frame()], 0, 1)))
    await asyncio.sleep(0.01)
    cancelled = True
    with pytest.raises(Cancelled):
        await task
    assert model.calls == pipeline.budget.calls == 0


async def test_one_call_failure_preserves_cause_and_requires_approval(settings):
    model = MockModel(responses=[SharError("PROVIDER_TRANSIENT", "HTTP 503", True, 503)])
    pipeline = Pipeline(settings.model_copy(update={"max_calls": 1}), model=model)
    with pytest.raises(SharError) as caught:
        await pipeline.infer(ModelRequest("temporal", "walk", [frame()], 0, 1))
    assert caught.value.message == "HTTP 503"
    assert pipeline.failure["can_retry"]
    assert not caught.value.retryable
    assert model.calls == 1


async def test_provider_delay_cannot_be_shortened_to_fit_deadline(settings):
    model = MockModel()
    pipeline = Pipeline(settings.model_copy(update={"job_timeout_sec": 0.1}), model=model)
    pipeline.cooldown = ProviderCooldown(settings)
    error = SharError("PROVIDER_RATE_LIMIT", "wait", True, 429)
    error.retry_after = 60
    pipeline.cooldown.record(error)
    with pytest.raises(SharError, match="remaining time"):
        await pipeline.infer(ModelRequest("temporal", "walk", [frame()], 0, 1))
    assert model.calls == 0


@pytest.mark.parametrize(
    "error",
    [
        httpx.ReadTimeout("timeout"),
        httpx.ConnectError("offline"),
        httpx.RemoteProtocolError("closed"),
    ],
)
def test_transport_errors_are_retryable(error):
    mapped = provider_error(error)
    assert mapped.retryable
    assert mapped.code in {"PROVIDER_TRANSIENT", "PROVIDER_TIMEOUT"}
