import hashlib
import json
import time
from contextlib import contextmanager

from sqlalchemy import and_, delete, func, insert, or_, select, update

from shar.errors import SharError
from shar.schemas import VideoAsset, uid
from shar.storage import models as m
from shar.storage.database import engine_for, initialize

TERMINAL = {"succeeded", "partial", "failed", "cancelled"}


class Repository:
    def __init__(self, settings):
        self.settings = settings
        self.engine = engine_for(settings.storage_root)
        initialize(self.engine)

    @contextmanager
    def transaction(self):
        with self.engine.connect() as db:
            db.exec_driver_sql("BEGIN IMMEDIATE")
            try:
                yield db
                db.commit()
            except BaseException:
                db.rollback()
                raise

    def add_video(self, asset):
        with self.engine.begin() as db:
            db.execute(
                insert(m.videos).values(
                    id=asset.id,
                    owner_id=asset.owner_id,
                    sha256=asset.sha256,
                    asset=asset.model_dump(mode="json"),
                    created_at=time.time(),
                    retention_until=time.time() + self.settings.retention_hours * 3600,
                )
            )

    def video(self, video_id, owner):
        with self.engine.connect() as db:
            row = (
                db.execute(
                    select(m.videos).where(m.videos.c.id == video_id, m.videos.c.owner_id == owner)
                )
                .mappings()
                .first()
            )
        if not row:
            raise SharError("NOT_FOUND", "Video not found", status=404)
        return VideoAsset.model_validate_json(json.dumps(row["asset"]))

    def submit(self, owner, request, config, key=None):
        digest = hashlib.sha256(
            json.dumps({"request": request, "config": config}, sort_keys=True).encode()
        ).hexdigest()
        now, job_id = time.time(), uid()
        with self.transaction() as db:
            db.execute(delete(m.idempotency_keys).where(m.idempotency_keys.c.expiry < now))
            if key:
                old = (
                    db.execute(
                        select(m.idempotency_keys).where(
                            m.idempotency_keys.c.owner_id == owner, m.idempotency_keys.c.key == key
                        )
                    )
                    .mappings()
                    .first()
                )
                if old:
                    if old["request_hash"] != digest:
                        raise SharError(
                            "IDEMPOTENCY_CONFLICT",
                            "Key was already used for a different request",
                            status=409,
                        )
                    return old["job_id"]
            video = db.execute(
                select(m.videos.c.id).where(
                    m.videos.c.id == request["video_id"], m.videos.c.owner_id == owner
                )
            ).first()
            if not video:
                raise SharError("NOT_FOUND", "Video not found", status=404)
            queued = db.execute(
                select(func.count())
                .select_from(m.jobs)
                .where(m.jobs.c.status.in_(["queued", "retry_wait", "running"]))
            ).scalar_one()
            if queued >= self.settings.max_queued_jobs:
                raise SharError("ADMISSION_LIMIT", "Job queue is full", True, 429)
            db.execute(
                insert(m.jobs).values(
                    id=job_id,
                    video_id=request["video_id"],
                    owner_id=owner,
                    request=request,
                    config=config,
                    status="queued",
                    stage="queued",
                    created_at=now,
                )
            )
            if key:
                db.execute(
                    insert(m.idempotency_keys).values(
                        owner_id=owner,
                        key=key,
                        request_hash=digest,
                        job_id=job_id,
                        expiry=now + 86400,
                    )
                )
            self._event(db, job_id, "queued")
        return job_id

    def job(self, job_id, owner=None):
        condition = m.jobs.c.id == job_id
        if owner is not None:
            condition = and_(condition, m.jobs.c.owner_id == owner)
        with self.engine.connect() as db:
            row = db.execute(select(m.jobs).where(condition)).mappings().first()
        if not row:
            raise SharError("NOT_FOUND", "Job not found", status=404)
        return dict(row)

    def _event(self, db, job_id, event, details=None):
        db.execute(
            insert(m.job_events).values(
                job_id=job_id, at=time.time(), event=event, details=details or {}
            )
        )

    def claim(self):
        now = time.time()
        with self.transaction() as db:
            expired = and_(m.jobs.c.status == "running", m.jobs.c.lease_until < now)
            db.execute(
                update(m.jobs)
                .where(expired, m.jobs.c.attempt >= self.settings.max_job_attempts)
                .values(status="failed", error_code="ATTEMPTS_EXHAUSTED", completed_at=now)
            )
            row = (
                db.execute(
                    select(m.jobs)
                    .where(
                        or_(
                            and_(
                                m.jobs.c.status.in_(["queued", "retry_wait"]),
                                m.jobs.c.next_attempt_at <= now,
                            ),
                            and_(expired, m.jobs.c.attempt < self.settings.max_job_attempts),
                        ),
                        m.jobs.c.cancel_requested.is_(False),
                    )
                    .order_by(m.jobs.c.created_at)
                    .limit(1)
                )
                .mappings()
                .first()
            )
            if not row:
                return None
            token = uid()
            db.execute(
                update(m.jobs)
                .where(m.jobs.c.id == row["id"])
                .values(
                    status="running",
                    stage="analysis",
                    lease_token=token,
                    lease_until=now + self.settings.lease_sec,
                    heartbeat_at=now,
                    started_at=row["started_at"] or now,
                    attempt=row["attempt"] + 1,
                )
            )
            self._event(db, row["id"], "claimed", {"attempt": row["attempt"] + 1})
            return {**dict(row), "lease_token": token, "attempt": row["attempt"] + 1}

    def fence(self, job_id, token):
        return and_(
            m.jobs.c.id == job_id,
            m.jobs.c.lease_token == token,
            m.jobs.c.status == "running",
            m.jobs.c.lease_until > time.time(),
        )

    def heartbeat(self, job_id, token):
        with self.engine.begin() as db:
            result = db.execute(
                update(m.jobs)
                .where(self.fence(job_id, token))
                .values(heartbeat_at=time.time(), lease_until=time.time() + self.settings.lease_sec)
            )
            return result.rowcount == 1

    def reserve_call(self, job_id, token, cap, estimated_cost=0.0, max_cost=None):
        with self.transaction() as db:
            if max_cost is not None:
                row = db.execute(
                    select(m.jobs.c.reserved_cost).where(self.fence(job_id, token))
                ).first()
                if not row or row[0] + estimated_cost > max_cost:
                    raise SharError("SPEND_BUDGET", "Durable spend budget exhausted")
            result = db.execute(
                update(m.jobs)
                .where(
                    self.fence(job_id, token),
                    m.jobs.c.calls < cap,
                    m.jobs.c.cancel_requested.is_(False),
                )
                .values(
                    calls=m.jobs.c.calls + 1, reserved_cost=m.jobs.c.reserved_cost + estimated_cost
                )
            )
            if result.rowcount != 1:
                raise SharError(
                    "CALL_BUDGET_OR_LEASE", "Call budget exhausted or worker lease lost"
                )

    def checkpoint(self, job_id, token, details):
        with self.transaction() as db:
            if not db.execute(select(m.jobs.c.id).where(self.fence(job_id, token))).first():
                raise SharError("LEASE_LOST", "Worker lease was lost")
            self._event(db, job_id, "checkpoint", details)

    def cancel(self, job_id, owner):
        with self.transaction() as db:
            row = (
                db.execute(select(m.jobs).where(m.jobs.c.id == job_id, m.jobs.c.owner_id == owner))
                .mappings()
                .first()
            )
            if not row:
                raise SharError("NOT_FOUND", "Job not found", status=404)
            if row["status"] not in TERMINAL:
                db.execute(
                    update(m.jobs)
                    .where(m.jobs.c.id == job_id)
                    .values(
                        cancel_requested=True,
                        status="cancelled",
                        stage="cancelled",
                        completed_at=time.time(),
                        lease_token=None,
                    )
                )
                self._event(db, job_id, "cancelled")

    def publish(self, job_id, token, report, uri, checksum):
        with self.transaction() as db:
            row = (
                db.execute(
                    select(m.jobs).where(
                        self.fence(job_id, token), m.jobs.c.cancel_requested.is_(False)
                    )
                )
                .mappings()
                .first()
            )
            if not row:
                raise SharError("LEASE_LOST", "Stale worker cannot publish")
            for track in report.tracks:
                track_id = job_id + ":" + track.actor_id
                db.execute(
                    insert(m.actor_tracks).values(
                        id=track_id, job_id=job_id, association_status=track.association_status
                    )
                )
                for sample in track.samples:
                    db.execute(
                        insert(m.spatial_detections).values(
                            track_id=track_id,
                            frame_id=sample.frame_id,
                            timestamp_sec=sample.timestamp_sec,
                            **sample.bbox.model_dump(),
                            source=sample.source,
                            payload=sample.model_dump(mode="json"),
                        )
                    )
            for action in report.predictions:
                action_id = job_id + ":" + action.id
                db.execute(
                    insert(m.action_predictions).values(
                        id=action_id,
                        job_id=job_id,
                        label=action.label,
                        start_sec=action.start_sec,
                        end_sec=action.end_sec,
                        score=action.confidence,
                        payload=action.model_dump(mode="json"),
                    )
                )
                for actor in action.actor_ids:
                    db.execute(
                        insert(m.action_actors).values(
                            action_id=action_id, track_id=job_id + ":" + actor
                        )
                    )
            db.execute(
                insert(m.processing_runs).values(
                    job_id=job_id,
                    payload={
                        "provenance": report.provenance,
                        "usage": report.usage.model_dump(),
                        "coverage": report.coverage.model_dump(),
                        "revisions": report.revisions,
                    },
                )
            )
            completion = db.execute(
                update(m.jobs)
                .where(self.fence(job_id, token))
                .values(
                    status=report.status,
                    stage="complete",
                    completed_at=time.time(),
                    result_uri=uri,
                    checksum=checksum,
                    lease_token=None,
                    lease_until=None,
                )
            )
            if completion.rowcount != 1:
                raise SharError("LEASE_LOST", "Lease expired during publication")
            self._event(db, job_id, report.status)

    def fail(self, job_id, token, error):
        with self.transaction() as db:
            row = db.execute(select(m.jobs).where(self.fence(job_id, token))).mappings().first()
            if not row:
                return
            retry = error.retryable and row["attempt"] < self.settings.max_job_attempts
            state = "retry_wait" if retry else "failed"
            db.execute(
                update(m.jobs)
                .where(self.fence(job_id, token))
                .values(
                    status=state,
                    stage=state,
                    error_code=error.code,
                    next_attempt_at=time.time() + min(60, 2 ** row["attempt"]),
                    completed_at=None if retry else time.time(),
                    lease_token=None,
                    lease_until=None,
                )
            )
            self._event(db, job_id, state, {"error_code": error.code})

    def start_stream(self, owner, source_name):
        session_id = uid()
        with self.transaction() as db:
            db.execute(
                insert(m.stream_sessions).values(
                    id=session_id,
                    owner_id=owner,
                    source_name=source_name,
                    status="running",
                    created_at=time.time(),
                    updated_at=time.time(),
                    retention_until=time.time() + self.settings.retention_hours * 3600,
                )
            )
        return session_id

    def stream_session(self, session_id, owner):
        with self.engine.connect() as db:
            row = (
                db.execute(
                    select(m.stream_sessions).where(
                        m.stream_sessions.c.id == session_id, m.stream_sessions.c.owner_id == owner
                    )
                )
                .mappings()
                .first()
            )
        if row is None:
            raise SharError("NOT_FOUND", "Stream session not found", status=404)
        return dict(row)

    def append_stream_event(self, session_id, owner, payload):
        with self.transaction() as db:
            row = db.execute(
                select(m.stream_sessions.c.id).where(
                    m.stream_sessions.c.id == session_id, m.stream_sessions.c.owner_id == owner
                )
            ).first()
            if row is None:
                raise SharError("NOT_FOUND", "Stream session not found", status=404)
            db.execute(insert(m.stream_events).values(session_id=session_id, payload=payload))
            terminal = payload.get("type") in {"session_end", "session_error"}
            db.execute(
                update(m.stream_sessions)
                .where(m.stream_sessions.c.id == session_id)
                .values(updated_at=time.time(), status="finished" if terminal else "running")
            )

    def read_stream_events(self, session_id, owner, after=0):
        self.stream_session(session_id, owner)
        with self.engine.connect() as db:
            rows = (
                db.execute(
                    select(m.stream_events)
                    .where(
                        m.stream_events.c.session_id == session_id,
                        m.stream_events.c.sequence > after,
                    )
                    .order_by(m.stream_events.c.sequence)
                    .limit(200)
                )
                .mappings()
                .all()
            )
        return [{"sequence": row["sequence"], **row["payload"]} for row in rows]
