from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Column,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    UniqueConstraint,
)

metadata = MetaData()
videos = Table(
    "videos",
    metadata,
    Column("id", String, primary_key=True),
    Column("owner_id", String, nullable=False),
    Column("sha256", String, nullable=False),
    Column("asset", JSON, nullable=False),
    Column("created_at", Float, nullable=False),
    Column("retention_until", Float, nullable=False),
)
Index("video_owner_hash", videos.c.owner_id, videos.c.sha256)
jobs = Table(
    "jobs",
    metadata,
    Column("id", String, primary_key=True),
    Column("video_id", ForeignKey("videos.id"), nullable=False),
    Column("owner_id", String, nullable=False),
    Column("request", JSON, nullable=False),
    Column("config", JSON, nullable=False),
    Column("status", String, nullable=False),
    Column("stage", String, nullable=False, default="queued"),
    Column("created_at", Float, nullable=False),
    Column("started_at", Float),
    Column("completed_at", Float),
    Column("next_attempt_at", Float, nullable=False, default=0),
    Column("attempt", Integer, nullable=False, default=0),
    Column("lease_token", String),
    Column("lease_until", Float),
    Column("heartbeat_at", Float),
    Column("cancel_requested", Boolean, nullable=False, default=False),
    Column("error_code", String),
    Column("result_uri", String),
    Column("checksum", String),
    Column("calls", Integer, nullable=False, default=0),
    Column("reserved_cost", Float, nullable=False, default=0.0),
    CheckConstraint(
        "status IN ('queued','running','retry_wait','succeeded','partial','failed','cancelled')"
    ),
)
Index("jobs_ready", jobs.c.status, jobs.c.next_attempt_at)
idempotency_keys = Table(
    "idempotency_keys",
    metadata,
    Column("owner_id", String, primary_key=True),
    Column("key", String, primary_key=True),
    Column("request_hash", String, nullable=False),
    Column("job_id", ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False),
    Column("expiry", Float, nullable=False),
)
action_predictions = Table(
    "action_predictions",
    metadata,
    Column("id", String, primary_key=True),
    Column("job_id", ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False),
    Column("label", String, nullable=False),
    Column("start_sec", Float, nullable=False),
    Column("end_sec", Float, nullable=False),
    Column("score", Float),
    Column("payload", JSON, nullable=False),
    CheckConstraint("start_sec >= 0 AND end_sec > start_sec"),
)
Index("predictions_job_label", action_predictions.c.job_id, action_predictions.c.label)
actor_tracks = Table(
    "actor_tracks",
    metadata,
    Column("id", String, primary_key=True),
    Column("job_id", ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False),
    Column("association_status", String, nullable=False),
)
action_actors = Table(
    "action_actors",
    metadata,
    Column("action_id", ForeignKey("action_predictions.id", ondelete="CASCADE"), primary_key=True),
    Column("track_id", ForeignKey("actor_tracks.id", ondelete="CASCADE"), primary_key=True),
)
spatial_detections = Table(
    "spatial_detections",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("track_id", ForeignKey("actor_tracks.id", ondelete="CASCADE"), nullable=False),
    Column("frame_id", String, nullable=False),
    Column("timestamp_sec", Float, nullable=False),
    Column("x", Float, nullable=False),
    Column("y", Float, nullable=False),
    Column("width", Float, nullable=False),
    Column("height", Float, nullable=False),
    Column("source", String, nullable=False),
    Column("payload", JSON, nullable=False),
    UniqueConstraint("track_id", "frame_id", "source"),
    CheckConstraint(
        "x >= 0 AND y >= 0 AND width > 0 AND height > 0 AND x + width <= 1 AND y + height <= 1"
    ),
)
processing_runs = Table(
    "processing_runs",
    metadata,
    Column("job_id", ForeignKey("jobs.id", ondelete="CASCADE"), primary_key=True),
    Column("payload", JSON, nullable=False),
)
job_events = Table(
    "job_events",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("job_id", ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False),
    Column("at", Float, nullable=False),
    Column("event", String, nullable=False),
    Column("details", JSON, nullable=False),
)

stream_sessions = Table(
    "stream_sessions",
    metadata,
    Column("id", String, primary_key=True),
    Column("owner_id", String, nullable=False),
    Column("source_name", String, nullable=False),
    Column("status", String, nullable=False),
    Column("created_at", Float, nullable=False),
    Column("updated_at", Float, nullable=False),
    Column("retention_until", Float, nullable=False),
)
stream_events = Table(
    "stream_events",
    metadata,
    Column("sequence", Integer, primary_key=True),
    Column("session_id", ForeignKey("stream_sessions.id", ondelete="CASCADE"), nullable=False),
    Column("payload", JSON, nullable=False),
)
Index("stream_session_sequence", stream_events.c.session_id, stream_events.c.sequence)
