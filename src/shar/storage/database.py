from pathlib import Path

from sqlalchemy import create_engine, event

from shar.storage.models import metadata


def engine_for(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    engine = create_engine(
        f"sqlite:///{root.resolve() / 'shar.sqlite'}",
        connect_args={"timeout": 10, "check_same_thread": False},
    )

    @event.listens_for(engine, "connect")
    def configure(connection, _):
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=10000")

    return engine


def initialize(engine):
    metadata.create_all(engine)
