import asyncio
import logging
import math
import os
import tempfile
import time
from dataclasses import replace
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import httpx

from shar.errors import SharError
from shar.inference import prompts
from shar.inference.multimodal.base import ModelCapabilities, ModelResponse
from shar.schemas import ModelPayload, Usage
from shar.telemetry import event


def provider_error(exc):
    """Keep actionable quota information without exposing provider response bodies."""
    code = getattr(exc, "code", None)
    body = getattr(exc, "details", {})
    body = body.get("error", body) if isinstance(body, dict) else {}
    body = body if isinstance(body, dict) else {}
    details = body.get("details", [])
    if not isinstance(details, list):
        details = []
    retry_after = 0.0
    headers = getattr(getattr(exc, "response", None), "headers", {}) or {}
    header = headers.get("Retry-After")
    if header is not None:
        try:
            retry_after = float(header)
        except (ValueError, TypeError):
            try:
                retry_after = parsedate_to_datetime(header).timestamp() - time.time()
            except (ValueError, TypeError, OverflowError):
                pass
    retry_after = max(0.0, retry_after) if math.isfinite(retry_after) else 0.0
    violations = []
    for detail in details:
        if not isinstance(detail, dict):
            continue
        if str(detail.get("@type", "")).endswith("google.rpc.RetryInfo"):
            delay = detail.get("retryDelay", "")
            try:
                seconds = float(delay[:-1]) if isinstance(delay, str) and delay.endswith("s") else 0
                if math.isfinite(seconds):
                    retry_after = max(retry_after, seconds)
            except ValueError:
                pass
        if str(detail.get("@type", "")).endswith("google.rpc.QuotaFailure"):
            items = detail.get("violations", [])
            if isinstance(items, list):
                violations.extend(item for item in items if isinstance(item, dict))
    retry_after = max(0.0, retry_after) if math.isfinite(retry_after) else 0.0
    daily = any(
        "perday" in str(v.get("quotaId", "")).lower().replace("_", "")
        or "perday" in str(v.get("quotaMetric", "")).lower().replace("_", "")
        for v in violations
    )
    unavailable = any(str(v.get("quotaValue", "")) == "0" for v in violations)
    network_error = isinstance(exc, (TimeoutError, ConnectionError, httpx.TransportError))
    # The async GenAI SDK may use aiohttp when its optional transport is installed.
    try:
        import aiohttp

        network_error = network_error or isinstance(exc, aiohttp.ClientConnectionError)
    except ImportError:
        pass
    transient = code in {429, 500, 502, 503, 504} or network_error
    error_code, message = {
        429: ("PROVIDER_RATE_LIMIT", "Gemini rate limit reached."),
        401: ("PROVIDER_AUTH", "Gemini rejected the server's API credentials."),
        403: ("PROVIDER_AUTH", "Gemini denied access. Check the server's API key and permissions."),
        404: ("PROVIDER_MODEL_NOT_FOUND", "The configured Gemini model is unavailable."),
        500: ("PROVIDER_TRANSIENT", "Gemini returned a server error (HTTP 500)."),
        502: ("PROVIDER_TRANSIENT", "Gemini's gateway failed (HTTP 502)."),
        503: ("PROVIDER_TRANSIENT", "Gemini is temporarily unavailable (HTTP 503)."),
        504: ("PROVIDER_TRANSIENT", "Gemini's gateway timed out (HTTP 504)."),
    }.get(
        code, ("PROVIDER_TRANSIENT" if transient else "PROVIDER_ERROR", "Provider request failed.")
    )
    if network_error:
        error_code = (
            "PROVIDER_TIMEOUT"
            if isinstance(exc, (TimeoutError, httpx.TimeoutException))
            else "PROVIDER_TRANSIENT"
        )
        message = (
            "The connection to Gemini timed out."
            if error_code == "PROVIDER_TIMEOUT"
            else "The connection to Gemini was interrupted."
        )
    if code == 429 and (daily or unavailable):
        transient = False
        error_code = "PROVIDER_DAILY_QUOTA" if daily else "PROVIDER_QUOTA_EXHAUSTED"
        message = (
            "Gemini's daily quota for this project and model is exhausted."
            if daily
            else "This project has no available Gemini quota for the configured model."
        )
        message += " Check usage and billing in Google AI Studio."
    error = SharError(
        error_code, message, transient, 429 if code == 429 else 503 if transient else 422
    )
    error.retry_after = retry_after
    return error


def provider_schema(task=None):
    """Adapt Gemini's response_schema subset; strict validation remains local."""

    def convert(value):
        if isinstance(value, list):
            return [convert(item) for item in value]
        if not isinstance(value, dict):
            return value
        result = {
            key: convert(item)
            for key, item in value.items()
            if key
            not in {
                "default",
                "title",
                "exclusiveMinimum",
                "exclusiveMaximum",
                "additionalProperties",
                # Large nested array/string limits can exceed Gemini's schema
                # complexity budget. Enforce these on the returned payload.
                "minItems",
                "maxItems",
                "minLength",
                "maxLength",
                "pattern",
            }
        }
        if "exclusiveMinimum" in value:
            result["minimum"] = value["exclusiveMinimum"]
        if "exclusiveMaximum" in value:
            result["maximum"] = value["exclusiveMaximum"]
        return result

    return convert(prompts.output_schema(task))


class GeminiModel:
    def __init__(self, settings, client=None):
        from google import genai
        from google.genai import types

        self.settings = settings
        if client is None and not os.environ.get("GEMINI_API_KEY"):
            raise SharError("PROVIDER_CONFIG", "GEMINI_API_KEY is required")
        self.client: Any = client or genai.Client(
            api_key=os.environ["GEMINI_API_KEY"],
            http_options=types.HttpOptions(
                timeout=int(settings.call_timeout_sec * 1000),
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )
        self.remote_files: set[str] = set()

    def capabilities(self):
        return ModelCapabilities(native_video=True, max_frames=self.settings.max_frames)

    def estimate_usage(self, request):
        return (
            len(request.frames) * self.settings.estimated_tokens_per_frame
            + len(request.prompt) // 4
            + self.settings.output_tokens
        )

    async def infer(self, request):
        from google.genai import types

        started = time.monotonic()
        contents = [types.Part.from_text(text=request.prompt)]
        remote = None
        temporary = None
        offset = 0.0
        try:
            if request.video_path:
                temporary = tempfile.TemporaryDirectory(prefix="shar-native-")
                clip = Path(temporary.name) / "clip.mp4"
                offset = request.start_sec
                process = await asyncio.create_subprocess_exec(
                    "ffmpeg",
                    "-y",
                    "-v",
                    "error",
                    "-ss",
                    str(offset),
                    "-i",
                    request.video_path,
                    "-t",
                    str(request.end_sec - offset),
                    "-map",
                    "0:v:0",
                    "-an",
                    "-vf",
                    "setpts=PTS-STARTPTS",
                    "-c:v",
                    "libx264",
                    str(clip),
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                try:
                    await asyncio.wait_for(process.wait(), self.settings.upload_timeout_sec)
                finally:
                    if process.returncode is None:
                        process.kill()
                        await process.wait()
                if process.returncode != 0:
                    raise SharError("CLIP_FAILED", "Unable to prepare silent native-video clip")
                relative = replace(
                    request,
                    frames=[
                        f.model_copy(update={"timestamp_sec": f.timestamp_sec - offset})
                        for f in request.frames
                    ],
                    start_sec=0.0,
                    end_sec=request.end_sec - offset,
                )
                contents = [
                    types.Part.from_text(
                        text=prompts.build(relative)
                        + "\nThis request contains one silent clip. Return clip-relative numeric times; frame IDs remain as supplied."
                    )
                ]
                async with asyncio.timeout(self.settings.upload_timeout_sec):
                    remote = await self.client.aio.files.upload(file=str(clip))
                    self.remote_files.add(remote.name)
                    while getattr(remote.state, "name", str(remote.state)) == "PROCESSING":
                        await asyncio.sleep(1)
                        remote = await self.client.aio.files.get(name=remote.name)
                    if getattr(remote.state, "name", str(remote.state)) != "ACTIVE":
                        raise SharError("UPLOAD_FAILED", "Provider could not process video")
                contents.append(
                    types.Part.from_uri(file_uri=remote.uri, mime_type=remote.mime_type)
                )
            if not request.video_path:
                for frame in request.frames:
                    contents.extend(
                        [
                            types.Part.from_text(
                                text=f"Frame {frame.frame_id}; index {frame.frame_index}; video time {frame.timestamp_sec:.9f} seconds"
                            ),
                            types.Part.from_bytes(data=frame.jpeg, mime_type="image/jpeg"),
                        ]
                    )
            response = await self.client.aio.models.generate_content(
                model=self.settings.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=provider_schema(request.task),
                    temperature=0,
                    max_output_tokens=self.settings.output_tokens,
                    thinking_config=types.ThinkingConfig(
                        thinking_level=types.ThinkingLevel(self.settings.thinking_level.upper())
                    )
                    if self.settings.thinking_level
                    else None,
                ),
            )
            usage = response.usage_metadata
            candidates = response.candidates or []
            finish = str(candidates[0].finish_reason) if candidates else "REFUSAL"
            payload = response.text or ""
            if request.video_path and payload:
                try:
                    decoded = ModelPayload.model_validate_json(payload)
                    for prediction in decoded.predictions:
                        prediction.start_sec += offset
                        prediction.end_sec += offset
                        for boundary in (prediction.start_boundary, prediction.end_boundary):
                            if boundary:
                                boundary.estimate += offset
                                boundary.lower += offset
                                boundary.upper += offset
                    for track in decoded.tracks:
                        for detection in track.samples:
                            detection.timestamp_sec += offset
                    payload = decoded.model_dump_json()
                except ValueError as exc:
                    raise SharError(
                        "INVALID_MODEL_OUTPUT", "Invalid native-video response"
                    ) from exc
            return ModelResponse(
                payload=payload,
                model=getattr(response, "model_version", None) or self.settings.model,
                usage=Usage(
                    model_calls=1,
                    input_tokens=getattr(usage, "prompt_token_count", None),
                    output_tokens=getattr(usage, "candidates_token_count", None),
                ),
                latency_sec=time.monotonic() - started,
                finish_reason=finish,
                provider_request_id=getattr(response, "response_id", None),
            )
        except SharError:
            raise
        except Exception as exc:
            error = provider_error(exc)
            event(
                "provider_failure",
                level=logging.WARNING,
                error_code=error.code,
                http_status=getattr(exc, "code", None),
                exception_type=type(exc).__name__,
                retry_after_sec=error.retry_after,
            )
            raise error from exc
        finally:
            if remote:
                try:
                    await asyncio.wait_for(self.client.aio.files.delete(name=remote.name), 10)
                    self.remote_files.discard(remote.name)
                except Exception:
                    event("remote_cleanup_deferred", owned_files=len(self.remote_files))
            if temporary:
                temporary.cleanup()

    async def close(self):
        for name in list(self.remote_files):
            try:
                await asyncio.wait_for(self.client.aio.files.delete(name=name), 10)
                self.remote_files.discard(name)
            except Exception:
                event("remote_cleanup_failed", owned_files=len(self.remote_files))
        await self.client.aio.aclose()
