import hashlib
from pathlib import Path
from typing import BinaryIO

from shar.config import Settings
from shar.errors import SharError
from shar.schemas import VideoAsset, uid
from shar.storage.artifacts import LocalArtifacts
from shar.video.reader import inspect_isolated


def ingest(source: BinaryIO, owner: str, settings: Settings) -> VideoAsset:
    store = LocalArtifacts(settings.storage_root)
    key = f"videos/{uid()}.media"
    path = store.path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    digest, size = hashlib.sha256(), 0
    try:
        with path.open("xb") as output:
            while chunk := source.read(1024 * 1024):
                size += len(chunk)
                if size > settings.max_upload_bytes:
                    raise SharError("VIDEO_TOO_LARGE", "Video exceeds upload limit", status=413)
                digest.update(chunk)
                output.write(chunk)
        metadata = inspect_isolated(path, settings)
        return VideoAsset(
            owner_id=owner,
            sha256=digest.hexdigest(),
            managed_uri=key,
            bytes=size,
            format="video/mp4",
            metadata=metadata,
        )
    except BaseException:
        path.unlink(missing_ok=True)
        path.with_suffix(".index.sqlite").unlink(missing_ok=True)
        raise


def ingest_path(path: Path, owner: str, settings: Settings) -> VideoAsset:
    with path.open("rb") as source:
        return ingest(source, owner, settings)


def ingest_url(url: str, owner: str, settings: Settings) -> VideoAsset:
    # Arbitrary remote fetching remains disabled, including redirects and private IPs.
    raise SharError("URL_DISABLED", "Remote URL ingestion is disabled; upload a video", status=422)
