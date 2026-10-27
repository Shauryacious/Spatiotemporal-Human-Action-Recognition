import hashlib
import json
import time
from pathlib import Path

from shar.storage.artifacts import LocalArtifacts


class InferenceCache:
    def __init__(self, root: Path, ttl: float):
        self.store, self.ttl = LocalArtifacts(root / "cache"), ttl

    @staticmethod
    def key(scope: str, video_hash: str, request, settings) -> str:
        config = settings.public()
        # Workers shrink this deadline as a job waits in the queue. It does not
        # change the inference request and must not invalidate reusable responses.
        for name in (
            "job_timeout_sec",
            "max_calls",
            "max_job_attempts",
            "provider_max_attempts",
            "require_retry_approval",
            "retry_base_sec",
            "retry_max_sec",
            "call_timeout_sec",
            "cache_ttl_sec",
        ):
            config.pop(name, None)
        value = {
            "scope": scope,
            "video": video_hash,
            "task": request.task,
            "frames": [(f.frame_id, f.timestamp_sec, f.image_hash) for f in request.frames],
            "domain": [request.start_sec, request.end_sec],
            "prompt": request.prompt,
            "context": request.context,
            "query": request.query,
            "config": config,
            "schema_version": "1.0",
            "cache_version": 3,
        }
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

    def get(self, key):
        try:
            value = json.loads(self.store.get(key + ".json"))
            if time.time() - value["created"] < self.ttl:
                return {"payload": value["payload"], "model": value["model"]}
        except (FileNotFoundError, ValueError, KeyError):
            pass
        return None

    def put(self, key, payload, model):
        self.store.put(
            key + ".json",
            json.dumps({"created": time.time(), "payload": payload, "model": model}).encode(),
        )
