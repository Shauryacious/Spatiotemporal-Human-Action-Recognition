from types import SimpleNamespace
from unittest.mock import AsyncMock

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


async def test_real_sdk_serializes_schema_over_fake_http(settings):
    import json

    import httpx
    from google import genai
    from google.genai import types

    seen = []

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
        request = ModelRequest(
            "temporal", "walk", [frame()], 0.0, 1.0, prompt="offline schema test"
        )
        response = await model.infer(request)
        assert not validate_response(response, request).predictions
        assert seen[0]["generationConfig"]["responseMimeType"] == "application/json"
        assert (
            "responseSchema" in seen[0]["generationConfig"]
            or "responseJsonSchema" in seen[0]["generationConfig"]
        )
        await model.close()
