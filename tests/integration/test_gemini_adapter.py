from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from conftest import frame

from shar.inference.multimodal.base import ModelRequest
from shar.inference.multimodal.gemini import GeminiModel
from shar.inference.validation import validate_response


def client_with_response(payload='{"predictions":[],"recognition":[],"tracks":[],"presence":[]}'):
    response = SimpleNamespace(
        text=payload,
        usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=5),
        candidates=[SimpleNamespace(finish_reason="STOP")],
        model_version="test-version",
        response_id="response-id",
    )
    return SimpleNamespace(
        aio=SimpleNamespace(
            models=SimpleNamespace(generate_content=AsyncMock(return_value=response)),
            files=SimpleNamespace(
                upload=AsyncMock(
                    return_value=SimpleNamespace(
                        name="files/owned",
                        uri="https://provider/file",
                        mime_type="video/mp4",
                        state=SimpleNamespace(name="ACTIVE"),
                    )
                ),
                delete=AsyncMock(),
                get=AsyncMock(),
            ),
            aclose=AsyncMock(),
        )
    )


async def test_sdk_frame_contract(settings):
    client = client_with_response()
    model = GeminiModel(settings, client=client)
    request = ModelRequest("temporal", "walk", [frame()], 0.0, 1.0, prompt="test")
    result = await model.infer(request)
    assert result.usage.input_tokens == 10 and result.model == "test-version"
    assert not validate_response(result, request).predictions
    call = client.aio.models.generate_content.call_args.kwargs
    assert call["config"].response_mime_type == "application/json"
    assert call["contents"][2].inline_data.mime_type == "image/jpeg"
    await model.close()
    client.aio.aclose.assert_awaited_once()


async def test_native_offsets_and_owned_cleanup(settings, media):
    client = client_with_response(
        '{"predictions":[{"label":"walk","start_sec":0.0,"end_sec":1.0,"evidence_frame_ids":["f0"]}]}'
    )
    model = GeminiModel(settings, client=client)
    request = ModelRequest(
        "temporal", "walk", [frame(0, 1.0)], 1.0, 2.0, video_path=str(media / "cfr.mp4")
    )
    result = await model.infer(request)
    prediction = validate_response(result, request).predictions[0]
    assert (prediction.start_sec, prediction.end_sec) == (1.0, 2.0)
    client.aio.files.delete.assert_awaited_once_with(name="files/owned")
    assert not model.remote_files
    await model.close()


@pytest.mark.parametrize("task", ["temporal", "spatial"])
async def test_real_sdk_serializes_schema_over_fake_http(settings, task):
    import json

    import httpx
    from google import genai
    from google.genai import types

    seen = []
    settings = settings.model_copy(update={"thinking_level": "low"})

    def handle(request):
        body = json.loads(request.content)
        seen.append(body)

        # Pydantic's extra="forbid" emits additionalProperties=False. The SDK
        # serializes it, but Gemini rejects it in the response_schema field.
        def assert_supported(value):
            if isinstance(value, dict):
                assert "additionalProperties" not in value
                assert "additional_properties" not in value
                assert "maxItems" not in value
                assert "maxLength" not in value
                for item in value.values():
                    assert_supported(item)
            elif isinstance(value, list):
                for item in value:
                    assert_supported(item)

        assert_supported(body["generationConfig"]["responseSchema"])
        if task == "spatial":
            assert set(body["generationConfig"]["responseSchema"]["properties"]) == {"tracks"}
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [
                                {
                                    "text": '{"predictions":[],"recognition":[],"tracks":[],"presence":[]}'
                                }
                            ],
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
                "modelVersion": "transport-test",
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        client = genai.Client(
            api_key="offline-fake-key",
            http_options=types.HttpOptions(
                httpx_async_client=http, retry_options=types.HttpRetryOptions(attempts=1)
            ),
        )
        model = GeminiModel(settings, client)
        request = ModelRequest(task, "walk", [frame()], 0.0, 1.0, prompt="offline schema test")
        response = await model.infer(request)
        assert not validate_response(response, request).predictions
        assert seen[0]["generationConfig"]["responseMimeType"] == "application/json"
        thinking = seen[0]["generationConfig"]["thinkingConfig"]
        assert thinking.get("thinkingLevel", thinking.get("thinking_level")) == "LOW"
        assert (
            "responseSchema" in seen[0]["generationConfig"]
            or "responseJsonSchema" in seen[0]["generationConfig"]
        )
        await model.close()


async def test_provider_failure_preserves_rate_limit_category(settings):
    import pytest
    from google.genai import errors

    from shar.errors import SharError

    client = client_with_response()
    client.aio.models.generate_content.side_effect = errors.ClientError(
        429, {"error": {"message": "Quota exceeded", "status": "RESOURCE_EXHAUSTED"}}
    )
    model = GeminiModel(settings, client)
    with pytest.raises(SharError) as caught:
        await model.infer(ModelRequest("temporal", "walk", [frame()], 0.0, 1.0))
    assert caught.value.code == "PROVIDER_RATE_LIMIT"
    assert caught.value.retryable
    await model.close()


def test_daily_quota_is_terminal_and_retains_structured_retry_delay():
    from google.genai import errors

    from shar.inference.multimodal.gemini import provider_error

    error = provider_error(
        errors.ClientError(
            429,
            {
                "error": {
                    "message": "sensitive provider details must not be exposed",
                    "status": "RESOURCE_EXHAUSTED",
                    "details": [
                        {
                            "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                            "violations": [
                                {
                                    "quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
                                    "quotaValue": "20",
                                }
                            ],
                        },
                        {
                            "@type": "type.googleapis.com/google.rpc.RetryInfo",
                            "retryDelay": "53008s",
                        },
                    ],
                }
            },
        )
    )
    assert error.code == "PROVIDER_DAILY_QUOTA"
    assert not error.retryable
    assert error.retry_after == 53008
    assert "daily quota" in error.message
    assert "sensitive" not in error.message


def test_minute_quota_honors_larger_of_header_and_structured_delay():
    import httpx
    from google.genai import errors

    from shar.inference.multimodal.gemini import provider_error

    error = provider_error(
        errors.ClientError(
            429,
            {
                "error": {
                    "details": [
                        {
                            "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                            "violations": [
                                {
                                    "quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier",
                                    "quotaValue": "10",
                                }
                            ],
                        },
                        {
                            "@type": "type.googleapis.com/google.rpc.RetryInfo",
                            "retryDelay": "32.5s",
                        },
                    ]
                }
            },
            response=httpx.Response(429, headers={"Retry-After": "20"}),
        )
    )
    assert error.code == "PROVIDER_RATE_LIMIT"
    assert error.retryable
    assert error.retry_after == 32.5


def test_zero_quota_does_not_retry():
    from google.genai import errors

    from shar.inference.multimodal.gemini import provider_error

    error = provider_error(
        errors.ClientError(
            429,
            {
                "error": {
                    "details": [
                        {
                            "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                            "violations": [
                                {
                                    "quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier",
                                    "quotaValue": "0",
                                }
                            ],
                        },
                    ]
                }
            },
        )
    )
    assert error.code == "PROVIDER_QUOTA_EXHAUSTED"
    assert not error.retryable
