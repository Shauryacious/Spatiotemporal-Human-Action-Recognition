import sqlite3
import time
from pathlib import Path

from shar.errors import SharError


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

    def reserve(self, tokens: int) -> int:
        now = time.time()
        with sqlite3.connect(self.path, timeout=10) as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("DELETE FROM reservations WHERE at < ?", (now - 86400,))
            minute = db.execute(
                "SELECT count(*), coalesce(sum(tokens),0) FROM reservations WHERE scope=? AND at>?",
                (self.scope, now - 60),
            ).fetchone()
            day = db.execute(
                "SELECT count(*) FROM reservations WHERE scope=?", (self.scope,)
            ).fetchone()[0]
            if day >= self.rpd:
                raise SharError("DAILY_QUOTA", "Daily provider quota exhausted")
            if minute[0] >= self.rpm or minute[1] + tokens > self.tpm:
                raise SharError("RATE_LIMIT", "Provider quota capacity unavailable", True, 429)
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
