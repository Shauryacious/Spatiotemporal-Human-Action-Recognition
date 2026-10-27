import time

import pytest
from sqlalchemy import update

from shar.errors import SharError
from shar.storage import models as m
from shar.storage.artifacts import LocalArtifacts
from shar.storage.repository import Repository


def submit(repo, asset, settings, key=None):
    return repo.submit(
        "local", {"video_id": asset.id, "prompt": "walk"}, settings.model_dump(mode="json"), key
    )


def test_idempotency_claim_fencing(settings, asset):
    settings = settings.model_copy(update={"require_retry_approval": False})
    repo = Repository(settings)
    repo.add_video(asset)
    a = submit(repo, asset, settings, "key")
    assert submit(repo, asset, settings, "key") == a
    with pytest.raises(SharError):
        repo.submit(
            "local",
            {"video_id": asset.id, "prompt": "run"},
            settings.model_dump(mode="json"),
            "key",
        )
    first = repo.claim()
    assert repo.claim() is None
    with repo.engine.begin() as db:
        db.execute(update(m.jobs).where(m.jobs.c.id == a).values(lease_until=time.time() - 1))
    second = repo.claim()
    assert second["lease_token"] != first["lease_token"]
    assert not repo.heartbeat(a, first["lease_token"])
    with pytest.raises(SharError):
        repo.reserve_call(a, first["lease_token"], 100)


def test_atomic_artifacts_paths(tmp_path):
    store = LocalArtifacts(tmp_path / "artifacts")
    store.put("safe/data", b"one")
    store.put("safe/data", b"two")
    assert store.get("safe/data") == b"two"
    with pytest.raises(SharError):
        store.get("../escape")
    (tmp_path / "artifacts/link").symlink_to(tmp_path)
    with pytest.raises(SharError):
        store.put("link/nope", b"bad")


def test_concurrent_claim_is_unique(settings, asset):
    from concurrent.futures import ThreadPoolExecutor

    repo = Repository(settings)
    repo.add_video(asset)
    submit(repo, asset, settings)
    with ThreadPoolExecutor(max_workers=2) as pool:
        claims = list(pool.map(lambda _: Repository(settings).claim(), range(2)))
    assert sum(c is not None for c in claims) == 1


def test_migration_empty_database(settings, monkeypatch):
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import inspect

    from shar.storage.database import engine_for

    monkeypatch.setenv("SHAR_STORAGE_ROOT", str(settings.storage_root))
    command.upgrade(Config("alembic.ini"), "head")
    tables = inspect(engine_for(settings.storage_root)).get_table_names()
    assert "jobs" in tables and "action_actors" in tables and "alembic_version" in tables


def test_spend_budget_survives_restart(settings, asset):
    repo = Repository(settings)
    repo.add_video(asset)
    job = submit(repo, asset, settings)
    claim = repo.claim()
    repo.reserve_call(job, claim["lease_token"], 100, 0.6, 1.0)
    restarted = Repository(settings)
    with pytest.raises(SharError, match="spend"):
        restarted.reserve_call(job, claim["lease_token"], 100, 0.6, 1.0)
    assert restarted.job(job)["calls"] == 1


def test_expiry_during_publication_rolls_back_all_rows(settings, asset, monkeypatch):
    from sqlalchemy import func, select

    from shar.schemas import ActionReport

    repo = Repository(settings)
    repo.add_video(asset)
    job = submit(repo, asset, settings)
    claim = repo.claim()
    real_fence = repo.fence
    calls = 0

    def fence(job_id, token):
        nonlocal calls
        calls += 1
        return real_fence(job_id, token) if calls == 1 else m.jobs.c.id == "expired"

    monkeypatch.setattr(repo, "fence", fence)
    report = ActionReport(job_id=job, video_id=asset.id, video=asset.metadata)
    with pytest.raises(SharError, match="expired during"):
        repo.publish(job, claim["lease_token"], report, "result.json", "checksum")
    with repo.engine.connect() as db:
        assert db.execute(select(func.count()).select_from(m.processing_runs)).scalar() == 0
    assert repo.job(job)["result_uri"] is None


def test_retention_reclaims_unregistered_cli_media(settings, asset):
    import os

    from shar.workers.cleanup import cleanup

    path = settings.storage_root / asset.managed_uri
    expired = time.time() - settings.retention_hours * 3600 - 1
    os.utime(path, (expired, expired))
    cleanup(settings)
    assert not path.exists()
    assert not path.with_suffix(".index.sqlite").exists()
