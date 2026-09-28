import json

from pydantic import ValidationError

from shar.errors import SharError
from shar.schemas import ModelPayload


def validate_response(response, request) -> ModelPayload:
    if response.finish_reason not in {"STOP", "FinishReason.STOP"}:
        raise SharError("MODEL_REFUSAL", "Provider did not complete a valid analysis")
    try:
        if isinstance(response.payload, str) and len(response.payload.encode()) > 2_000_000:
            raise ValueError("response exceeds payload limit")
        raw = (
            response.payload.model_dump(mode="json")
            if isinstance(response.payload, ModelPayload)
            else response.payload
        )
        payload = ModelPayload.model_validate_json(raw if isinstance(raw, str) else json.dumps(raw))
        frames = {f.frame_id: f for f in request.frames}
        known_actors = {t.actor_id for t in payload.tracks} | set(
            request.context.get("prediction", {}).get("actor_ids", [])
        )
        if len({p.id for p in payload.predictions}) != len(payload.predictions):
            raise ValueError("duplicate prediction IDs")
        if len({t.actor_id for t in payload.tracks}) != len(payload.tracks):
            raise ValueError("duplicate actor IDs")
        for action in payload.predictions:
            if not set(action.actor_ids) <= known_actors:
                raise ValueError("unknown actor reference")
            if not request.start_sec <= action.start_sec < action.end_sec <= request.end_sec + 1e-6:
                raise ValueError("interval outside request domain")
            if not action.evidence_frame_ids or not set(action.evidence_frame_ids) <= frames.keys():
                raise ValueError("missing or unknown evidence")
            if not any(
                action.start_sec <= frames[f].timestamp_sec < action.end_sec
                for f in action.evidence_frame_ids
            ):
                raise ValueError("no evidence inside proposed interval")
            for b in (action.start_boundary, action.end_boundary):
                if b and not request.start_sec <= b.lower <= b.upper <= request.end_sec + 1e-6:
                    raise ValueError("boundary outside request")
        for label in payload.recognition:
            if not label.evidence_frame_ids or not set(label.evidence_frame_ids) <= frames.keys():
                raise ValueError("missing or unknown recognition evidence")
        for track in payload.tracks:
            for detection in track.samples:
                f = frames.get(detection.frame_id)
                if (
                    f is None
                    or abs(f.timestamp_sec - detection.timestamp_sec) > 1e-6
                    or f.frame_index != detection.frame_index
                ):
                    raise ValueError("spatial observation does not match supplied frame")
                if not detection.observed:
                    raise ValueError("inference must return observed boxes only")
        if any(p.frame_id not in frames for p in payload.presence):
            raise ValueError("unknown presence frame")
        if request.task == "recognize" and payload.predictions:
            raise ValueError("recognition must not fabricate temporal intervals")
        return payload
    except (ValidationError, ValueError, TypeError) as exc:
        raise SharError(
            "INVALID_MODEL_OUTPUT", "Model response failed schema or evidence validation"
        ) from exc
