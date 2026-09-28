import time

from sqlalchemy import delete, select

from shar.storage import models as m
from shar.storage.artifacts import LocalArtifacts
from shar.storage.repository import TERMINAL, Repository


def cleanup(settings):
    repo, store, removed = Repository(settings), LocalArtifacts(settings.storage_root), 0
    with repo.transaction() as db:
        expired_streams = (
            db.execute(
                select(m.stream_sessions.c.id).where(
                    m.stream_sessions.c.retention_until < time.time(),
                    m.stream_sessions.c.updated_at < time.time() - settings.job_timeout_sec,
                )
            )
            .scalars()
            .all()
        )
        for session_id in expired_streams:
            store.delete(f"streams/{session_id}/preview.jpg")
            db.execute(delete(m.stream_sessions).where(m.stream_sessions.c.id == session_id))
        assets = (
            db.execute(select(m.videos).where(m.videos.c.retention_until < time.time()))
            .mappings()
            .all()
        )
        for asset in assets:
            jobs = (
                db.execute(select(m.jobs).where(m.jobs.c.video_id == asset["id"])).mappings().all()
            )
            if any(j["status"] not in TERMINAL for j in jobs):
                continue
            for job in jobs:
                if job["result_uri"]:
                    store.delete(job["result_uri"])
            store.path(asset["asset"]["managed_uri"]).with_suffix(".index.sqlite").unlink(
                missing_ok=True
            )
            store.delete(asset["asset"]["managed_uri"])
            db.execute(delete(m.jobs).where(m.jobs.c.video_id == asset["id"]))
            db.execute(delete(m.videos).where(m.videos.c.id == asset["id"]))
            removed += 1
        referenced_videos = {
            row["managed_uri"] for row in db.execute(select(m.videos.c.asset)).scalars()
        }
        referenced = set(
            db.execute(
                select(m.jobs.c.result_uri).where(m.jobs.c.result_uri.is_not(None))
            ).scalars()
        )
    for path in (settings.storage_root / "videos").glob("*.media"):
        key = str(path.relative_to(settings.storage_root))
        if (
            key not in referenced_videos
            and time.time() - path.stat().st_mtime > settings.retention_hours * 3600
        ):
            store.path(key).with_suffix(".index.sqlite").unlink(missing_ok=True)
            store.delete(key)
    for folder in ("results", "cache"):
        for path in (settings.storage_root / folder).rglob("*.json"):
            age = time.time() - path.stat().st_mtime
            key = str(path.relative_to(settings.storage_root))
            if (folder == "cache" and age > settings.cache_ttl_sec) or (
                folder == "results" and key not in referenced and age > settings.lease_sec * 2
            ):
                store.delete(key)
    return removed
