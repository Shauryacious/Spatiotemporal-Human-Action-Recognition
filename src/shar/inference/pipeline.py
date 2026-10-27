import asyncio
import random
import time

from shar.config import Settings
from shar.errors import Cancelled, SharError
from shar.inference import prompts
from shar.inference.multimodal.base import ModelRequest, ModelResponse
from shar.inference.multimodal.cache import InferenceCache
from shar.inference.multimodal.limits import Budget, SharedLimiter
from shar.inference.multimodal.mock import MockModel
from shar.inference.refinement import refine
from shar.inference.spatial.localizer import MLLMLocalizer
from shar.inference.temporal.merge import nms, stitch
from shar.inference.temporal.proposals import candidates
from shar.inference.validation import validate_response
from shar.schemas import ActionPrediction, ActionReport, BoundaryEstimate, Coverage, Usage
from shar.telemetry import event
from shar.video.sampling import sample, windows


def make_model(settings):
    if settings.provider == "mock":
        return MockModel()
    from shar.inference.multimodal.gemini import GeminiModel

    try:
        return GeminiModel(settings)
    except ImportError as exc:
        raise SharError("PROVIDER_CONFIG", "Install the gemini extra to use this provider") from exc


class Pipeline:
    def __init__(
        self, settings: Settings, model=None, cancelled=None, reserve=None, checkpoint=None
    ):
        self.settings = settings
        self.model = model or make_model(settings)
        self.cancelled = cancelled or (lambda: False)
        self.reserve, self.checkpoint = reserve, checkpoint
        self.cache = InferenceCache(settings.storage_root, settings.cache_ttl_sec)
        self.limiter = (
            SharedLimiter(
                settings.storage_root,
                settings.model,
                settings.rpm or 1,
                settings.tpm or 1,
                settings.rpd or 1,
            )
            if settings.provider == "gemini"
            else None
        )
        self.usage = Usage()
        self.scope, self.video_hash = "", ""
        self.budget = Budget(
            settings.max_calls, time.monotonic() + settings.job_timeout_sec, reserve
        )
        self.estimated_spend = 0.0
        self.resolved_models: set[str] = set()
        self.manifests: list[dict] = []
        self.unknown_usage: set[str] = set()

    def check(self):
        if self.cancelled():
            raise Cancelled()
        if time.monotonic() >= self.budget.deadline:
            raise SharError("DEADLINE", "Job deadline exceeded")

    async def _infer_guarded(self, request):
        task = asyncio.create_task(self.model.infer(request))
        try:
            while not task.done():
                self.check()
                await asyncio.wait({task}, timeout=0.1)
            self.check()
            return await task
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    async def infer(self, request):
        self.check()
        request.prompt = prompts.build(request)
        if (
            len(request.frames) > self.settings.max_frames
            or sum(len(f.jpeg) for f in request.frames) > self.settings.max_request_bytes
        ):
            raise SharError("REQUEST_BUDGET", "Frame request exceeds configured size budget")
        self.manifests.append(
            {
                "task": request.task,
                "domain": [request.start_sec, request.end_sec],
                "frames": [f.model_dump(mode="json", exclude={"jpeg"}) for f in request.frames],
            }
        )
        key = self.cache.key(self.scope, self.video_hash, request, self.settings)
        cached = self.cache.get(key)
        if cached is not None:
            self.usage.cache_hits += 1
            if self.resolved_models and cached["model"] not in self.resolved_models:
                raise SharError("MODEL_DRIFT", "Resolved model version changed within the job")
            self.resolved_models.add(cached["model"])
            return validate_response(ModelResponse(cached["payload"], cached["model"]), request)
        corrected = False
        for attempt in range(3):
            self.check()
            self.budget.check()
            cost = 0.0
            estimated_tokens = self.model.estimate_usage(request)
            if self.settings.max_cost is not None:
                assert self.settings.input_price_per_million is not None
                assert self.settings.output_price_per_million is not None
                cost = (
                    estimated_tokens * self.settings.input_price_per_million / 1e6
                    + self.settings.output_tokens * self.settings.output_price_per_million / 1e6
                )
                if self.estimated_spend + cost > self.settings.max_cost:
                    raise SharError("SPEND_BUDGET", "Estimated spend budget exhausted")
                self.estimated_spend += cost
            self.budget.reserve(cost)
            self.usage.model_calls += 1
            reservation = self.limiter.reserve(estimated_tokens) if self.limiter else None
            started = time.monotonic()
            try:
                timeout = min(
                    self.settings.call_timeout_sec, self.budget.deadline - time.monotonic()
                )
                response = await asyncio.wait_for(self._infer_guarded(request), timeout)
                self.check()
                if self.resolved_models and response.model not in self.resolved_models:
                    raise SharError("MODEL_DRIFT", "Resolved model version changed within the job")
                self.resolved_models.add(response.model)
                for name in ("input_tokens", "output_tokens"):
                    value = getattr(response.usage, name)
                    if value is not None:
                        setattr(self.usage, name, (getattr(self.usage, name) or 0) + value)
                    else:
                        self.unknown_usage.add(name)
                if self.limiter and reservation:
                    actual = response.usage.input_tokens
                    self.limiter.reconcile(
                        reservation,
                        actual + (response.usage.output_tokens or 0)
                        if actual is not None
                        else None,
                    )
                payload = validate_response(response, request)
                self.cache.put(key, payload.model_dump(mode="json"), response.model)
                event(
                    "inference",
                    task=request.task,
                    calls=self.usage.model_calls,
                    elapsed_sec=time.monotonic() - started,
                )
                return payload
            except TimeoutError as exc:
                self.unknown_usage.update({"input_tokens", "output_tokens"})
                error = SharError("PROVIDER_TIMEOUT", "Model call timed out", True, 503)
                if attempt == 2:
                    raise error from exc
            except SharError as exc:
                error = exc
                if exc.code != "INVALID_MODEL_OUTPUT":
                    self.unknown_usage.update({"input_tokens", "output_tokens"})
                if exc.code == "INVALID_MODEL_OUTPUT" and not corrected:
                    corrected = True
                    request.prompt += "\nCORRECTION: Previous output failed validation. Follow the schema and exact evidence manifest."
                elif not exc.retryable or attempt == 2:
                    raise
            await asyncio.sleep(
                min(
                    max(getattr(error, "retry_after", 0), 0.2 * 2**attempt + random.random() * 0.1),
                    max(0.0, self.budget.deadline - time.monotonic()),
                )
            )
        raise SharError("INVALID_MODEL_OUTPUT", "Model output remained invalid after retries")

    async def run(
        self, asset, query: str, job_id: str | None = None, interval=None, track_only=False
    ) -> ActionReport:
        started = time.monotonic()
        self.scope, self.video_hash = asset.owner_id, asset.sha256
        path = self.settings.storage_root / asset.managed_uri
        coverage = Coverage()
        predictions, recognition, tracks, warnings, revisions = [], [], [], [], []
        last_error = None
        start, end = interval or (0.0, asset.metadata.duration_sec)
        if not 0 <= start < end <= asset.metadata.duration_sec:
            raise SharError("INVALID_INTERVAL", "Requested interval exceeds video")
        try:
            for local_start, local_end in windows(
                end - start, self.settings.window_sec, self.settings.overlap_sec
            ):
                self.check()
                a, b = start + local_start, start + local_end
                try:
                    frames = await asyncio.to_thread(
                        sample, path, asset.metadata, a, b, self.settings.coarse_fps, self.settings
                    )
                    if not frames:
                        raise SharError("NO_FRAMES", "No usable frames in window")
                    if track_only:
                        predictions.append(
                            ActionPrediction(
                                label=query[:300],
                                start_sec=a,
                                end_sec=b,
                                evidence_frame_ids=[f.frame_id for f in frames],
                                spatial_status="unlocalized",
                                confidence_kind="user_supplied_interval",
                            )
                        )
                        coverage.analyzed_intervals.append((a, b))
                        continue
                    request = ModelRequest(
                        task="temporal" if self.settings.enable_temporal else "recognize",
                        query=query,
                        frames=frames,
                        start_sec=a,
                        end_sec=b,
                    )
                    if self.settings.input_strategy == "native_video":
                        if not self.model.capabilities().native_video:
                            raise SharError(
                                "NATIVE_VIDEO_CONFIG",
                                "Selected adapter does not support native video",
                            )
                        request.video_path = str(path)
                    payload = await self.infer(request)
                    # Namespace provider-local IDs per window; identity does not carry across
                    # independent requests without evidence-based association.
                    namespace = f"w{a:.6f}:"
                    for track in payload.tracks:
                        old_id = track.actor_id
                        track.actor_id = namespace + old_id
                        for prediction in payload.predictions:
                            prediction.actor_ids = [
                                track.actor_id if actor == old_id else actor
                                for actor in prediction.actor_ids
                            ]
                    tracks.extend(payload.tracks)
                    times = [a, *[f.timestamp_sec for f in frames], b]
                    for prediction in payload.predictions:
                        prediction.id = namespace + prediction.id
                        for field, value in (
                            ("start_boundary", prediction.start_sec),
                            ("end_boundary", prediction.end_sec),
                        ):
                            if getattr(prediction, field) is None:
                                setattr(
                                    prediction,
                                    field,
                                    BoundaryEstimate(
                                        estimate=value,
                                        lower=max(t for t in times if t <= value),
                                        upper=min(t for t in times if t >= value),
                                        truncated=value in {a, b},
                                    ),
                                )
                    predictions.extend(
                        candidates(
                            payload, self.settings.proposal_threshold, self.settings.max_candidates
                        )
                    )
                    recognition.extend(payload.recognition)
                    coverage.analyzed_intervals.append((a, b))
                    if self.checkpoint:
                        self.checkpoint({"window": [a, b], "cache_keyed": True})
                except SharError as exc:
                    if isinstance(exc, Cancelled):
                        raise
                    last_error = exc
                    coverage.failed_intervals.append((a, b))
                    warnings.append(exc.code)
                    if exc.code in {
                        "PROVIDER_CONFIG",
                        "MODEL_REFUSAL",
                        "PROVIDER_ERROR",
                        "NATIVE_VIDEO_CONFIG",
                        "MODEL_DRIFT",
                    }:
                        raise
            if not coverage.analyzed_intervals:
                if last_error is not None:
                    raise last_error
                raise SharError("NO_COVERAGE", "No window was successfully analyzed")
            if self.settings.enable_stitching:
                predictions = stitch(predictions)
            predictions = nms(predictions, self.settings.nms_tiou)
            if len(predictions) > self.settings.max_candidates:
                warnings.append("CANDIDATE_LIMIT")
                predictions = predictions[: self.settings.max_candidates]
            localizer = MLLMLocalizer(self.infer, query) if self.settings.enable_spatial else None
            refined = []
            for candidate in predictions:
                self.check()
                actor_tracks = [t for t in tracks if t.actor_id in candidate.actor_ids]

                async def dense(current, iteration=0):
                    # Keep dense boundary neighborhoods even for very long actions.
                    parts = []
                    capped = self.settings.model_copy(
                        update={"max_frames": max(2, self.settings.max_frames // 2)}
                    )
                    for boundary in (current.start_sec, current.end_sec):
                        lo, hi = max(start, boundary - 1), min(end, boundary + 1)
                        if lo < hi:
                            parts.extend(
                                await asyncio.to_thread(
                                    sample,
                                    path,
                                    asset.metadata,
                                    lo,
                                    hi,
                                    self.settings.fine_fps * (iteration + 1),
                                    capped,
                                )
                            )
                    return sorted(
                        {f.frame_id: f for f in parts}.values(), key=lambda f: f.timestamp_sec
                    )[: self.settings.max_frames]

                boundary_rounds = 0
                if (
                    self.settings.enable_boundary_search
                    and self.settings.boundary_calls
                    and not track_only
                ):
                    search_settings = self.settings.model_copy(
                        update={"max_rounds": self.settings.boundary_calls}
                    )
                    candidate, _, history, reason = await refine(
                        candidate, [], dense, self.infer, None, search_settings, self.budget
                    )
                    boundary_rounds = len(history)
                    revisions.append(
                        {
                            "prediction_id": candidate.id,
                            "stage": "boundary_search",
                            "stop_reason": reason,
                            "history": history,
                        }
                    )
                    if reason not in {
                        "max_iterations",
                        "converged",
                        "repeated_state",
                        "no_new_evidence",
                    }:
                        warnings.append("BOUNDARY_" + reason.upper())
                if self.settings.enable_refinement:
                    candidate, actor_tracks, history, reason = await refine(
                        candidate,
                        actor_tracks,
                        dense,
                        self.infer,
                        None,
                        self.settings.model_copy(
                            update={
                                "boundary_calls": max(
                                    0, self.settings.boundary_calls - boundary_rounds
                                )
                            }
                        ),
                        self.budget,
                    )
                    revisions.append(
                        {"prediction_id": candidate.id, "stop_reason": reason, "history": history}
                    )
                    if reason not in {
                        "max_iterations",
                        "converged",
                        "repeated_state",
                        "no_new_evidence",
                    }:
                        warnings.append("REFINEMENT_" + reason.upper())
                if localizer:
                    try:
                        # Sample the whole final interval, not just boundary neighborhoods.
                        spatial_frames = await asyncio.to_thread(
                            sample,
                            path,
                            asset.metadata,
                            candidate.start_sec,
                            candidate.end_sec,
                            self.settings.fine_fps,
                            self.settings,
                        )
                        actor_tracks = await localizer.localize(candidate, spatial_frames)
                    except SharError as exc:
                        if isinstance(exc, Cancelled):
                            raise
                        warnings.append(exc.code)
                    candidate = candidate.model_copy(
                        update={
                            "actor_ids": [t.actor_id for t in actor_tracks],
                            "spatial_status": "partial" if actor_tracks else "unlocalized",
                        }
                    )
                if localizer and not actor_tracks:
                    warnings.append("SPATIAL_UNLOCALIZED")
                refined.append(candidate)
                tracks.extend(actor_tracks)
            for field in self.unknown_usage:
                setattr(self.usage, field, None)
            if (
                self.settings.input_price_per_million is not None
                and self.settings.output_price_per_million is not None
                and self.usage.input_tokens is not None
                and self.usage.output_tokens is not None
            ):
                self.usage.estimated_cost = (
                    self.usage.input_tokens * self.settings.input_price_per_million
                    + self.usage.output_tokens * self.settings.output_price_per_million
                ) / 1e6
            labels: dict = {}
            for label in recognition:
                if label.label not in labels or (label.confidence or -1) > (
                    labels[label.label].confidence or -1
                ):
                    labels[label.label] = label
            refined = nms(refined, self.settings.nms_tiou)
            referenced = {actor for prediction in refined for actor in prediction.actor_ids}
            tracks = list({t.actor_id: t for t in tracks if t.actor_id in referenced}.values())
            kwargs = {"job_id": job_id} if job_id else {}
            report = ActionReport(
                video_id=asset.id,
                video=asset.metadata,
                predictions=refined,
                tracks=tracks,
                recognition=list(labels.values()),
                coverage=coverage,
                usage=self.usage,
                warnings=sorted(set(warnings)),
                revisions=revisions,
                status="partial" if warnings or coverage.failed_intervals else "succeeded",
                provenance={
                    "provider": self.settings.provider,
                    "model": self.settings.model,
                    "resolved_models": sorted(self.resolved_models),
                    "prompt_version": prompts.version_hash(),
                    "config_hash": self.settings.fingerprint(),
                    "config": self.settings.public(),
                    "video_sha256": asset.sha256,
                    "elapsed_sec": time.monotonic() - started,
                    "code_version": "0.1.0",
                    "sampling_manifests": self.manifests,
                    "mock_semantics": self.settings.provider == "mock",
                    "region_source": "user_supplied" if track_only else "model",
                },
                **kwargs,
            )
            return report
        finally:
            await self.model.close()
