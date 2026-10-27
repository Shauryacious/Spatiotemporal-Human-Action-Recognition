import asyncio
import hashlib
import json
import random
import time

from shar.config import Settings
from shar.errors import Cancelled, SharError
from shar.inference import prompts
from shar.inference.multimodal.base import ModelRequest, ModelResponse
from shar.inference.multimodal.cache import InferenceCache
from shar.inference.multimodal.limits import Budget, ProviderCooldown, SharedLimiter, provider_scope
from shar.inference.multimodal.mock import MockModel
from shar.inference.refinement import refine
from shar.inference.spatial.localizer import MLLMLocalizer
from shar.inference.temporal.merge import nms, stitch
from shar.inference.temporal.proposals import candidates
from shar.inference.validation import validate_response
from shar.schemas import ActionPrediction, ActionReport, BoundaryEstimate, Coverage, Usage
from shar.storage.artifacts import LocalArtifacts
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
        self,
        settings: Settings,
        model=None,
        cancelled=None,
        reserve=None,
        checkpoint=None,
        resume_root_id=None,
        resume_required=False,
    ):
        self.settings = settings
        self.model = model or make_model(settings)
        self.cancelled = cancelled or (lambda: False)
        self.reserve, self.checkpoint = reserve, checkpoint
        self.cache = InferenceCache(settings.storage_root, settings.cache_ttl_sec)
        self.limiter = (
            SharedLimiter(
                settings.storage_root,
                provider_scope(settings),
                settings.rpm or 1,
                settings.tpm or 1,
                settings.rpd or 1,
            )
            if settings.provider == "gemini"
            else None
        )
        self.cooldown = ProviderCooldown(settings) if settings.provider == "gemini" else None
        self.usage = Usage()
        self.scope, self.video_hash = "", ""
        self.budget = Budget(
            settings.max_calls, time.monotonic() + settings.job_timeout_sec, reserve
        )
        self.estimated_spend = 0.0
        self.resolved_models: set[str] = set()
        self.manifests: list[dict] = []
        self.unknown_usage: set[str] = set()
        self.provider_blocked: SharError | None = None
        self.failure: dict | None = None
        self.resume_root_id = resume_root_id
        self.resume_required = resume_required
        self.response_index = None
        self.saved = LocalArtifacts(settings.storage_root)
        self.last_call_key = None

    def response_index_key(self):
        return f"run-responses/{self.resume_root_id}/index.json"

    def saved_response(self, key):
        if not self.resume_root_id:
            return None
        try:
            if self.response_index is None:
                try:
                    self.response_index = json.loads(self.saved.get(self.response_index_key()))
                except FileNotFoundError:
                    if self.resume_required:
                        raise
                    self.response_index = {}
                    self.saved.put(self.response_index_key(), b"{}")
            if not isinstance(self.response_index, dict):
                raise ValueError("invalid response index")
            if key not in self.response_index:
                return None
            raw = self.saved.get(f"run-responses/{self.resume_root_id}/{key}.json")
            if hashlib.sha256(raw).hexdigest() != self.response_index[key]:
                raise ValueError("saved response checksum mismatch")
            return json.loads(raw)
        except (FileNotFoundError, ValueError, TypeError) as exc:
            raise SharError(
                "SAVED_RESPONSE_UNAVAILABLE",
                "A completed response is missing or damaged. No new request was sent to regenerate it. Restore the saved run data before continuing.",
            ) from exc

    def save_response(self, key, value):
        if self.resume_root_id:
            checksum = self.saved.put(
                f"run-responses/{self.resume_root_id}/{key}.json", json.dumps(value).encode()
            )
            self.response_index[key] = checksum
            self.saved.put(self.response_index_key(), json.dumps(self.response_index).encode())

    def require_decision(self, request, error):
        if self.failure is None:
            self.failure = {
                "stage": {
                    "temporal": "analysis",
                    "recognize": "analysis",
                    "refine": "refinement",
                    "spatial": "localization",
                }.get(request.task, request.task),
                "task": request.task,
                "error_code": error.code,
                "message": error.message,
                "retry_at": time.time() + error.retry_after if error.retry_after else None,
                "can_retry": error.code
                in {
                    "PROVIDER_TRANSIENT",
                    "PROVIDER_TIMEOUT",
                    "PROVIDER_RATE_LIMIT",
                    "INVALID_MODEL_OUTPUT",
                    "INVALID_REFINEMENT",
                    "CALL_BUDGET",
                    "CALL_BUDGET_OR_LEASE",
                    "DEADLINE",
                    "DAILY_QUOTA",
                },
            }
            self.progress(
                "decision",
                "Stopped. No more model calls will be sent without your approval.",
                failure=self.failure,
            )
        error.retryable = False
        self.provider_blocked = error

    async def infer(self, request):
        try:
            return await self._infer(request)
        except SharError as error:
            if self.settings.require_retry_approval and not isinstance(error, Cancelled):
                self.require_decision(request, error)
            raise

    def refinement_decision(self, reason):
        if (
            not self.settings.require_retry_approval
            or self.failure
            or reason
            in {
                "max_iterations",
                "converged",
                "repeated_state",
                "no_new_evidence",
            }
        ):
            return
        if reason == "invalid_refinement":
            # Structurally valid output can still lack evidence for its boundary move.
            # Do not replay that rejected proposal on an approved retry.
            if self.last_call_key:
                self.cache.store.delete(self.last_call_key + ".json")
                if self.resume_root_id:
                    self.saved.delete(
                        f"run-responses/{self.resume_root_id}/{self.last_call_key}.json"
                    )
                    self.response_index.pop(self.last_call_key, None)
                    self.saved.put(
                        self.response_index_key(), json.dumps(self.response_index).encode()
                    )
            message = "The model's proposed boundary change was not supported by the supplied frames. The previous interval was kept."
        else:
            message = f"Boundary refinement stopped: {reason.replace('_', ' ')}. The previous interval was kept."
        self.require_decision(
            ModelRequest("refine", "", [], 0.0, 1.0),
            SharError(reason.upper(), message),
        )

    def progress(self, stage, message, **details):
        if self.checkpoint:
            self.checkpoint({"stage": stage, "message": message, **details})

    def check(self):
        if self.cancelled():
            raise Cancelled()
        if time.monotonic() >= self.budget.deadline:
            raise SharError("DEADLINE", "Job deadline exceeded")

    async def wait_to_retry(self, task, error, delay):
        if delay >= self.budget.deadline - time.monotonic():
            blocked = SharError(
                error.code,
                error.message + " The retry delay exceeds this job's remaining time.",
                False,
                error.status,
            )
            blocked.retry_after = error.retry_after
            self.provider_blocked = blocked
            raise blocked from error
        self.progress(
            "model",
            f"Waiting {delay:.1f}s before {task}: {error.message}",
            error_code=error.code,
            retry_after_sec=delay,
            retry_at=time.time() + delay,
        )
        retry_at = time.monotonic() + delay
        while time.monotonic() < retry_at:
            self.check()
            await asyncio.sleep(min(0.1, max(0.0, retry_at - time.monotonic())))
        self.check()

    async def reserve_capacity(self, request, estimated_tokens):
        # Waiting for capacity is not a provider attempt and consumes no call budget.
        while True:
            self.check()
            self.budget.check()
            try:
                if self.cooldown:
                    self.cooldown.check()
                return self.limiter.reserve(estimated_tokens) if self.limiter else None
            except SharError as error:
                if not error.retryable or error.retry_after <= 0:
                    raise
                await self.wait_to_retry(request.task, error, error.retry_after)

    def exhausted_error(self, error, attempts):
        reason = (
            f"The run's model-call limit ({self.budget.max_calls}) is exhausted."
            if self.budget.calls >= self.budget.max_calls
            else "Automatic retries for this request are exhausted."
        )
        exhausted = SharError(
            error.code,
            f"{error.message} {attempts} provider attempt(s) failed. {reason}",
            False,
            error.status,
        )
        exhausted.retry_after = error.retry_after
        return exhausted

    async def _infer_guarded(self, request):
        task = asyncio.create_task(self.model.infer(request))
        started = last_update = time.monotonic()
        try:
            while not task.done():
                self.check()
                now = time.monotonic()
                if now - last_update >= 5:
                    self.progress(
                        "model",
                        f"Waiting for {request.task} model response "
                        f"({now - started:.0f}s elapsed; "
                        f"{self.settings.call_timeout_sec:g}s call timeout)",
                    )
                    last_update = now
                await asyncio.wait({task}, timeout=0.1)
            self.check()
            return await task
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    async def _infer(self, request):
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
        self.last_call_key = key
        cached = self.saved_response(key)
        cached = cached or self.cache.get(key)
        if cached is not None:
            self.progress("model", f"Using cached {request.task} response")
            self.usage.cache_hits += 1
            if self.resolved_models and cached["model"] not in self.resolved_models:
                raise SharError("MODEL_DRIFT", "Resolved model version changed within the job")
            self.resolved_models.add(cached["model"])
            self.save_response(key, cached)
            return validate_response(ModelResponse(cached["payload"], cached["model"]), request)
        if self.provider_blocked:
            raise self.provider_blocked
        corrected = False
        self.budget.check()
        attempts = min(
            1 if self.settings.require_retry_approval else self.settings.provider_max_attempts,
            self.budget.max_calls - self.budget.calls,
        )
        for attempt in range(attempts):
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
            started = time.monotonic()
            reservation = await self.reserve_capacity(request, estimated_tokens)
            try:
                self.budget.reserve(cost)
                self.estimated_spend += cost
                self.usage.model_calls += 1
                self.progress(
                    "model",
                    (
                        f"Sending {request.task} request · new call {self.usage.model_calls}/{self.settings.max_calls}"
                        if self.settings.require_retry_approval
                        else f"Waiting for {request.task} model response (attempt {attempt + 1}/{attempts})"
                    ),
                )
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
                try:
                    payload = validate_response(response, request)
                except SharError as exc:
                    # Retain rejected output locally so it can be investigated without
                    # spending another provider call to reproduce it.
                    self.cache.store.put(
                        key + ".invalid.json",
                        json.dumps(
                            {
                                "payload": response.payload,
                                "model": response.model,
                                "finish_reason": response.finish_reason,
                                "error_code": exc.code,
                            },
                            default=str,
                        ).encode(),
                    )
                    raise
                self.cache.put(key, payload.model_dump(mode="json"), response.model)
                self.save_response(
                    key, {"payload": payload.model_dump(mode="json"), "model": response.model}
                )
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
                if self.settings.require_retry_approval:
                    raise error from exc
                if attempt == attempts - 1:
                    raise self.exhausted_error(error, attempt + 1) from exc
            except SharError as exc:
                error = exc
                if self.cooldown and exc.code.startswith("PROVIDER_"):
                    self.cooldown.record(exc)
                if exc.code in {"PROVIDER_DAILY_QUOTA", "PROVIDER_QUOTA_EXHAUSTED"}:
                    self.provider_blocked = exc
                if exc.code != "INVALID_MODEL_OUTPUT":
                    self.unknown_usage.update({"input_tokens", "output_tokens"})
                if self.settings.require_retry_approval:
                    raise
                if exc.code == "INVALID_MODEL_OUTPUT" and not corrected and attempt < attempts - 1:
                    corrected = True
                    request.prompt += "\nCORRECTION: Previous output failed validation. Follow the schema and exact evidence manifest."
                elif not exc.retryable:
                    raise
                elif attempt == attempts - 1:
                    # This call already exhausted its retries. Retrying the entire
                    # job repeats successful windows and multiplies provider calls.
                    raise self.exhausted_error(exc, attempt + 1) from exc
            delay = max(
                error.retry_after,
                min(self.settings.retry_max_sec, self.settings.retry_base_sec * 2**attempt)
                * random.uniform(1, 1.25),
            )
            await self.wait_to_retry(request.task, error, delay)
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
            segments = list(
                windows(end - start, self.settings.window_sec, self.settings.overlap_sec)
            )
            for index, (local_start, local_end) in enumerate(segments):
                self.check()
                a, b = start + local_start, start + local_end
                try:
                    self.progress(
                        "analysis",
                        f"Sampling window {index + 1}/{len(segments)} ({a:.1f}–{b:.1f}s)",
                        completed=index,
                        total=len(segments),
                    )
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
                        self.progress(
                            "analysis", "Window sampled", completed=index + 1, total=len(segments)
                        )
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
                    self.progress(
                        "analysis",
                        f"Window {index + 1}/{len(segments)} analyzed",
                        completed=index + 1,
                        total=len(segments),
                    )
                except SharError as exc:
                    if isinstance(exc, Cancelled):
                        raise
                    last_error = exc
                    coverage.failed_intervals.append((a, b))
                    warnings.append(exc.code)
                    self.progress(
                        "analysis",
                        f"Window {index + 1} failed: {exc.code}",
                        completed=index + 1,
                        total=len(segments),
                    )
                    if self.settings.require_retry_approval:
                        coverage.failed_intervals.extend(
                            (start + lo, start + hi) for lo, hi in segments[index + 1 :]
                        )
                        break
                    if exc.code in {
                        "PROVIDER_CONFIG",
                        "MODEL_REFUSAL",
                        "PROVIDER_ERROR",
                        "PROVIDER_AUTH",
                        "PROVIDER_MODEL_NOT_FOUND",
                        "PROVIDER_DAILY_QUOTA",
                        "PROVIDER_QUOTA_EXHAUSTED",
                        "NATIVE_VIDEO_CONFIG",
                        "MODEL_DRIFT",
                    }:
                        raise
            if not coverage.analyzed_intervals:
                if last_error is not None:
                    raise last_error
                raise SharError("NO_COVERAGE", "No window was successfully analyzed")
            self.progress("merging", "Merging overlapping action proposals")
            if self.settings.enable_stitching:
                predictions = stitch(predictions)
            predictions = nms(predictions, self.settings.nms_tiou)
            if len(predictions) > self.settings.max_candidates:
                warnings.append("CANDIDATE_LIMIT")
                predictions = predictions[: self.settings.max_candidates]
            localizer = MLLMLocalizer(self.infer, query) if self.settings.enable_spatial else None
            refined = []
            for candidate_index, candidate in enumerate(predictions):
                if self.settings.enable_refinement or (
                    self.settings.enable_boundary_search and not track_only
                ):
                    self.progress(
                        "refinement",
                        f"Processing action {candidate_index + 1}/{len(predictions)}",
                        completed=candidate_index,
                        total=len(predictions),
                    )
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
                    and not self.failure
                ):
                    self.progress("refinement", "Searching action boundaries")
                    search_settings = self.settings.model_copy(
                        update={"max_rounds": self.settings.boundary_calls}
                    )
                    candidate, _, history, reason = await refine(
                        candidate, [], dense, self.infer, None, search_settings, self.budget
                    )
                    boundary_rounds = len(history)
                    self.refinement_decision(reason)
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
                if self.settings.enable_refinement and not self.failure:
                    self.progress("refinement", "Refining temporal boundaries")
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
                    self.refinement_decision(reason)
                    if reason not in {
                        "max_iterations",
                        "converged",
                        "repeated_state",
                        "no_new_evidence",
                    }:
                        warnings.append("REFINEMENT_" + reason.upper())
                if localizer and not self.failure:
                    self.progress("localization", "Sampling action frames and localizing actors")
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
            self.progress("finalizing", "Validating and assembling results")
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
                    "failure": self.failure,
                },
                **kwargs,
            )
            return report
        finally:
            await self.model.close()
