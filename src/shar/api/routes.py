import hashlib
import json
import time
from typing import Annotated

from fastapi import APIRouter, Depends, Header, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import text
from starlette.concurrency import run_in_threadpool

from shar.api.auth import owner
from shar.config import Settings
from shar.errors import SharError
from shar.schemas import ActionReport, JobStatus
from shar.video.ingest import ingest

router = APIRouter()
Owner = Annotated[str, Depends(owner)]


class Submit(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    video_id: str | None = None
    video_url: str | None = None
    prompt: str = Field(min_length=1, max_length=4000)
    enable_temporal: bool = True
    enable_spatial: bool = False
    enable_refinement: bool = True
    config_profile: str = "research-default"
    interval: tuple[float, float] | None = None
    prediction_id: str | None = None
    source_job_id: str | None = None

    @model_validator(mode="after")
    def inputs(self):
        if bool(self.video_id) == bool(self.video_url):
            raise ValueError("Provide exactly one input source")
        if self.interval and not 0 <= self.interval[0] < self.interval[1]:
            raise ValueError("Invalid interval")
        return self


def submit(body, request, identity, key, task="analysis"):
    if body.video_url:
        raise SharError("URL_DISABLED", "Remote URLs are disabled; upload a video")
    if body.config_profile != "research-default":
        raise SharError("INVALID_PROFILE", "Unknown configuration profile")
    asset = request.app.state.repo.video(body.video_id, identity)
    if body.interval and body.interval[1] > asset.metadata.duration_sec:
        raise SharError("INVALID_INTERVAL", "Interval exceeds video duration")
    values = request.app.state.settings.model_dump()
    values.update(
        {
            name: getattr(body, name)
            for name in ("enable_temporal", "enable_spatial", "enable_refinement")
        }
    )
    try:
        settings = Settings.model_validate(values)
    except ValueError as exc:
        raise SharError(
            "INVALID_OPTIONS", "Spatial and refinement options require temporal localization"
        ) from exc
    job_id = request.app.state.repo.submit(
        identity,
        {**body.model_dump(mode="json"), "task": task},
        settings.model_dump(mode="json"),
        key,
    )
    job = request.app.state.repo.job(job_id, identity)
    return JSONResponse(
        {"job_id": job_id, "status": job["status"]},
        status_code=202,
        headers={"Location": f"/jobs/{job_id}"},
    )


@router.post("/videos", status_code=201)
async def upload(request: Request, file: UploadFile, identity: Owner):
    try:
        asset = await run_in_threadpool(ingest, file.file, identity, request.app.state.settings)
        try:
            request.app.state.repo.add_video(asset)
        except BaseException:
            request.app.state.store.delete(asset.managed_uri)
            raise
        return {"video_id": asset.id, "metadata": asset.metadata.model_dump()}
    finally:
        await file.close()


@router.post("/jobs")
def jobs(
    body: Submit,
    request: Request,
    identity: Owner,
    idempotency_key: Annotated[str | None, Header(max_length=200)] = None,
):
    return submit(body, request, identity, idempotency_key)


@router.post("/localize")
def localize(
    body: Submit,
    request: Request,
    identity: Owner,
    idempotency_key: Annotated[str | None, Header(max_length=200)] = None,
):
    body.enable_temporal = True
    return submit(body, request, identity, idempotency_key)


@router.post("/analyze")
def analyze(
    body: Submit,
    request: Request,
    identity: Owner,
    idempotency_key: Annotated[str | None, Header(max_length=200)] = None,
):
    if "enable_temporal" not in body.model_fields_set:
        body.enable_temporal = False
        body.enable_refinement = False
    return submit(body, request, identity, idempotency_key)


@router.post("/track")
def track(
    body: Submit,
    request: Request,
    identity: Owner,
    idempotency_key: Annotated[str | None, Header(max_length=200)] = None,
):
    if body.prediction_id:
        if not body.source_job_id:
            raise SharError("INVALID_OPTIONS", "Existing prediction requires source_job_id")
        previous = request.app.state.repo.job(body.source_job_id, identity)
        if previous["video_id"] != body.video_id or not previous["result_uri"]:
            raise SharError("INVALID_OPTIONS", "Source result must belong to this video")
        report = ActionReport.model_validate_json(
            request.app.state.store.get(previous["result_uri"])
        )
        prediction = next((p for p in report.predictions if p.id == body.prediction_id), None)
        if prediction is None:
            raise SharError("NOT_FOUND", "Prediction not found", status=404)
        body.interval = (prediction.start_sec, prediction.end_sec)
    if not body.interval:
        raise SharError("INVALID_OPTIONS", "Tracking requires an interval or existing prediction")
    body.enable_temporal, body.enable_spatial, body.enable_refinement = True, True, False
    return submit(body, request, identity, idempotency_key, task="track")


@router.get("/jobs/{job_id}", response_model=JobStatus)
def job_status(job_id: str, request: Request, identity: Owner):
    job = request.app.state.repo.job(job_id, identity)
    return {
        key: job[key]
        for key in (
            "id",
            "status",
            "stage",
            "created_at",
            "started_at",
            "completed_at",
            "attempt",
            "cancel_requested",
            "error_code",
            "calls",
            "next_attempt_at",
        )
    }


@router.get("/results/{job_id}")
def results(job_id: str, request: Request, identity: Owner):
    job = request.app.state.repo.job(job_id, identity)
    if not job["result_uri"]:
        raise SharError("RESULT_NOT_READY", "No completed result is available", status=409)
    raw = request.app.state.store.get(job["result_uri"])
    if hashlib.sha256(raw).hexdigest() != job["checksum"]:
        raise SharError(
            "ARTIFACT_INTEGRITY", "Stored result failed integrity validation", status=503
        )
    return json.loads(raw)


@router.post("/jobs/{job_id}/cancel", status_code=202)
def cancel(job_id: str, request: Request, identity: Owner):
    request.app.state.repo.cancel(job_id, identity)
    return job_status(job_id, request, identity)


@router.get("/videos/{video_id}/content")
def content(video_id: str, request: Request, identity: Owner):
    asset = request.app.state.repo.video(video_id, identity)
    return FileResponse(
        request.app.state.store.path(asset.managed_uri),
        media_type=asset.format,
        headers={"X-Content-Type-Options": "nosniff"},
    )


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/ready")
def ready(request: Request):
    try:
        if request.app.state.settings.provider == "gemini":
            import importlib.util
            import os

            if (
                not os.environ.get("GEMINI_API_KEY")
                or importlib.util.find_spec("google.genai") is None
            ):
                raise ValueError("provider configuration unavailable")
        with request.app.state.repo.engine.connect() as db:
            db.execute(text("SELECT 1"))
        age = time.time() - float(request.app.state.store.get("worker-heartbeat"))
        if age > request.app.state.settings.lease_sec:
            raise ValueError("worker unavailable")
    except Exception as exc:
        raise SharError("NOT_READY", "Check server persistence, the running shar worker, and provider configuration", True, 503) from exc
    return {"status": "ready"}


@router.get("/connection")
def connection(request: Request, identity: Owner):
    """Check app authentication and worker readiness without calling the provider."""
    ready(request)
    settings = request.app.state.settings
    return {"status": "ready", "provider": settings.provider, "model": settings.model}


@router.get("/streams/{session_id}")
def stream_status(session_id: str, request: Request, identity: Owner):
    return request.app.state.repo.stream_session(session_id, identity)


@router.get("/streams/{session_id}/events")
def stream_events(session_id: str, request: Request, identity: Owner, after: int = 0):
    if after < 0:
        raise SharError("INVALID_CURSOR", "Event cursor must be nonnegative")
    return {"events": request.app.state.repo.read_stream_events(session_id, identity, after)}


@router.get("/streams/{session_id}/preview")
def stream_preview(session_id: str, request: Request, identity: Owner):
    request.app.state.repo.stream_session(session_id, identity)
    path = request.app.state.store.path(f"streams/{session_id}/preview.jpg")
    if not path.exists():
        raise SharError("NOT_FOUND", "No preview observation is available", status=404)
    return FileResponse(path, media_type="image/jpeg")
