"""
Media storage backends.

``MediaStorage`` is the seam: the pipeline persists bytes through it and gets
back a URI to record in metadata. The default ``LocalMediaStorage`` writes to
disk; swap in an S3/GCS implementation with the same interface and nothing else
changes. Raw bytes live here and nowhere near memory.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# Map MIME type to a file extension for human-readable stored objects.
_EXT: dict[str, str] = {"image/png": "png", "image/jpeg": "jpg"}


@runtime_checkable
class MediaStorage(Protocol):
    """Persist raw bytes and return a stable URI reference."""

    async def save(self, *, data: bytes, mime_type: str) -> str: ...


class LocalMediaStorage:
    """Filesystem storage. Content-addressed (sha256) so re-uploads dedupe."""

    def __init__(self, base_dir: str) -> None:
        self._base = Path(base_dir)

    async def save(self, *, data: bytes, mime_type: str) -> str:
        digest = hashlib.sha256(data).hexdigest()
        ext = _EXT.get(mime_type, "bin")
        path = self._base / f"{digest}.{ext}"
        await asyncio.to_thread(self._write, path, data)
        return path.resolve().as_uri()

    @staticmethod
    def _write(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():  # content-addressed → identical bytes already stored
            path.write_bytes(data)


__all__ = ["MediaStorage", "LocalMediaStorage"]
