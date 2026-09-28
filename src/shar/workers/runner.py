import asyncio
import contextlib
import time

from shar.config import Settings
from shar.errors import SharError
from shar.inference.pipeline import Pipeline
from shar.storage.artifacts import LocalArtifacts
from shar.storage.repository import Repository


class Worker:
    def __init__(self, settings, model_factory=None):
        self.settings, self.repo = settings, Repository(settings)
        self.store = LocalArtifacts(settings.storage_root)
        self.model_factory = model_factory

    async def once(self):
        self.store.put("worker-heartbeat", str(time.time()).encode())
        job = self.repo.claim()
        if not job:
            return False
        job_id, token = job["id"], job["lease_token"]
        lost = False

        async def heartbeat():
            nonlocal lost
            while True:
                await asyncio.sleep(self.settings.lease_sec / 3)
                if not self.repo.heartbeat(job_id, token):
                    lost = True
                    return
                self.store.put("worker-heartbeat", str(time.time()).encode())

        pulse = asyncio.create_task(heartbeat())

        def cancelled():
            row = self.repo.job(job_id)
            return lost or row["cancel_requested"] or row["lease_token"] != token

        try:
            settings = Settings.model_validate(job["config"])
            remaining = settings.job_timeout_sec - (time.time() - job["created_at"])
            if remaining <= 0:
                raise SharError("DEADLINE", "Job deadline expired before execution")
            settings = settings.model_copy(update={"job_timeout_sec": remaining})
            model = self.model_factory(settings) if self.model_factory else None
            pipeline = Pipeline(
                settings,
                model=model,
                cancelled=cancelled,
                reserve=lambda cost: self.repo.reserve_call(
                    job_id, token, settings.max_calls, cost, settings.max_cost
                ),
                checkpoint=lambda details: self.repo.checkpoint(job_id, token, details),
            )
            pipeline.budget.calls = job["calls"]
            asset = self.repo.video(job["video_id"], job["owner_id"])
            async with asyncio.timeout(remaining):
                report = await pipeline.run(
                    asset,
                    job["request"]["prompt"],
                    job_id,
                    job["request"].get("interval"),
                    track_only=job["request"].get("task") == "track",
                )
            report.usage.model_calls = self.repo.job(job_id)["calls"]
            if job["calls"]:
                report.usage.input_tokens = None
                report.usage.output_tokens = None
                report.usage.estimated_cost = None
                report.provenance["prior_attempt_usage"] = (
                    "unavailable; durable call and reserved-cost totals retained"
                )
            report.provenance["reserved_cost"] = self.repo.job(job_id)["reserved_cost"]
            uri = f"results/{job_id}/{token}.json"
            checksum = self.store.put(uri, report.model_dump_json(indent=2).encode())
            self.repo.publish(job_id, token, report, uri, checksum)
        except TimeoutError:
            self.repo.fail(job_id, token, SharError("DEADLINE", "Job deadline exceeded"))
        except SharError as error:
            self.repo.fail(job_id, token, error)
        except Exception:
            self.repo.fail(job_id, token, SharError("INTERNAL_ERROR", "Worker failed unexpectedly"))
            raise
        finally:
            pulse.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pulse
        return True

    async def run(self):
        while True:
            if not await self.once():
                await asyncio.sleep(1)
