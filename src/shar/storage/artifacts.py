import hashlib
import os
import tempfile
from pathlib import Path
from typing import Protocol

from shar.errors import SharError


class ArtifactStore(Protocol):
    def put(self, key: str, data: bytes) -> str: ...
    def get(self, key: str) -> bytes: ...
    def delete(self, key: str) -> None: ...


class LocalArtifacts:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, key: str) -> Path:
        candidate = self.root / key
        if Path(key).is_absolute() or ".." in Path(key).parts:
            raise SharError("UNSAFE_PATH", "Invalid artifact key")
        if any(p.is_symlink() for p in (candidate, *candidate.parents) if p != self.root.parent):
            raise SharError("UNSAFE_PATH", "Symbolic links are not allowed")
        if not candidate.resolve().is_relative_to(self.root):
            raise SharError("UNSAFE_PATH", "Artifact is outside managed storage")
        return candidate

    def put(self, key: str, data: bytes) -> str:
        path = self.path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        finally:
            Path(tmp).unlink(missing_ok=True)
        return hashlib.sha256(data).hexdigest()

    def get(self, key: str) -> bytes:
        return self.path(key).read_bytes()

    def delete(self, key: str) -> None:
        self.path(key).unlink(missing_ok=True)
