import hashlib
import math
import os
import sqlite3
import time
from pathlib import Path

from shar.errors import SharError


def provider_scope(settings):
    # A replacement credential may belong to a different project. Never store the key.
    credential = hashlib.sha256(os.environ.get("GEMINI_API_KEY", "").encode()).hexdigest()
    return f"{settings.model}:{credential}"


class ProviderCooldown:
    """Share provider-requested pauses across API processes, workers, and jobs."""

    def __init__(self, settings):
        settings.storage_root.mkdir(parents=True, exist_ok=True)
        self.path = settings.storage_root / "quotas.sqlite"
        self.scope = provider_scope(settings)
        with sqlite3.connect(self.path) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS provider_cooldowns "
                "(scope TEXT PRIMARY KEY, until REAL, code TEXT, message TEXT, status INTEGER)"
            )

    def record(self, error):
        if error.retry_after <= 0 or not math.isfinite(error.retry_after):
            return
        with sqlite3.connect(self.path) as db:
            db.execute(
                "INSERT INTO provider_cooldowns VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(scope) DO UPDATE SET until=excluded.until, code=excluded.code, "
                "message=excluded.message, status=excluded.status "
                "WHERE excluded.until > provider_cooldowns.until",
                (
                    self.scope,
                    time.time() + error.retry_after,
                    error.code,
                    error.message,
                    error.status,
                ),
            )

    def check(self):
        with sqlite3.connect(self.path) as db:
            row = db.execute(
                "SELECT until, code, message, status FROM provider_cooldowns WHERE scope=?",
                (self.scope,),
            ).fetchone()
        remaining = row[0] - time.time() if row else 0
        if remaining > 0:
            error = SharError(
                row[1],
                row[2],
                row[1] in {"PROVIDER_RATE_LIMIT", "PROVIDER_TRANSIENT", "PROVIDER_TIMEOUT"},
                row[3],
            )
            error.retry_after = remaining
            raise error

    def clear(self):
        with sqlite3.connect(self.path) as db:
            db.execute("DELETE FROM provider_cooldowns WHERE scope=?", (self.scope,))


class Budget:
    def __init__(self, max_calls: int, deadline: float, reserve=None):
        self.max_calls, self.deadline, self.calls = max_calls, deadline, 0
        self.persist_reserve = reserve

    def check(self):
        if time.monotonic() >= self.deadline:
            raise SharError("DEADLINE", "Job deadline exceeded")
        if self.calls >= self.max_calls:
            raise SharError("CALL_BUDGET", "Job model-call budget exhausted")

    def reserve(self, estimated_cost=0.0):
        self.check()
        if self.persist_reserve:
            self.persist_reserve(estimated_cost)
        self.calls += 1


class SharedLimiter:
    """Cross-process account/model quota reservation, durable across worker restarts.

    Reservations are conservative: failed/unknown usage is never refunded.
    """

    def __init__(self, root: Path, scope: str, rpm: int, tpm: int, rpd: int):
        root.mkdir(parents=True, exist_ok=True)
        self.path, self.scope = root / "quotas.sqlite", scope
        self.rpm, self.tpm, self.rpd = rpm, tpm, rpd
        with sqlite3.connect(self.path) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS reservations (id INTEGER PRIMARY KEY, scope TEXT, at REAL, tokens INTEGER)"
            )
            db.execute("CREATE INDEX IF NOT EXISTS quota_scope_time ON reservations(scope, at)")

    def daily_retry_after(self, db=None, now=None) -> float:
        """Time until rolling daily capacity returns, including lowered limits."""
        if now is None:
            now = time.time()
        if db is None:
            with sqlite3.connect(self.path) as connection:
                return self.daily_retry_after(connection, now)
        row = db.execute(
            "SELECT at FROM reservations WHERE scope=? AND at>? "
            "ORDER BY at DESC LIMIT 1 OFFSET ?",
            (self.scope, now - 86400, self.rpd - 1),
        ).fetchone()
        return max(0.0, row[0] + 86400 - now) if row else 0.0

    def reserve(self, tokens: int) -> int:
        now = time.time()
        if tokens > self.tpm:
            raise SharError(
                "REQUEST_BUDGET", "This request exceeds the configured tokens-per-minute limit."
            )
        with sqlite3.connect(self.path, timeout=10) as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("DELETE FROM reservations WHERE at <= ?", (now - 86400,))
            minute = db.execute(
                "SELECT count(*), coalesce(sum(tokens),0) FROM reservations WHERE scope=? AND at>?",
                (self.scope, now - 60),
            ).fetchone()
            day = db.execute(
                "SELECT count(*) FROM reservations WHERE scope=?", (self.scope,)
            ).fetchone()[0]
            if day >= self.rpd:
                error = SharError(
                    "DAILY_QUOTA",
                    f"The configured local daily request limit ({self.rpd} requests in a "
                    "rolling 24-hour window) is reached. Wait until the displayed retry time "
                    "and approve a retry to reuse saved responses. SHAR_RPD controls this "
                    "local cap; change it only to match your verified provider allowance.",
                    status=429,
                )
                error.retry_after = self.daily_retry_after(db, now)
                raise error
            if minute[0] >= self.rpm or minute[1] + tokens > self.tpm:
                rows = db.execute(
                    "SELECT at, tokens FROM reservations WHERE scope=? AND at>? ORDER BY at",
                    (self.scope, now - 60),
                ).fetchall()
                count, used = minute
                retry_at = now + 60
                for at, reserved in rows:
                    count, used = count - 1, used - reserved
                    if count < self.rpm and used + tokens <= self.tpm:
                        retry_at = at + 60
                        break
                error = SharError("RATE_LIMIT", "Waiting for local request capacity.", True, 429)
                error.retry_after = max(0.01, retry_at - now)
                raise error
            cursor = db.execute(
                "INSERT INTO reservations(scope,at,tokens) VALUES (?,?,?)",
                (self.scope, now, tokens),
            )
            assert cursor.lastrowid is not None
            return cursor.lastrowid

    def reconcile(self, reservation: int, actual: int | None):
        if actual is not None:
            with sqlite3.connect(self.path) as db:
                db.execute(
                    "UPDATE reservations SET tokens=? WHERE id=? AND scope=?",
                    (actual, reservation, self.scope),
                )
