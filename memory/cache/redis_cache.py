"""
RedisStateCache — hot snapshot cache for patient_state.

Cache key: patient_state:{patient_id}
TTL:       300 seconds by default (RuntimeConfig.MEMORY_CACHE_TTL_SEC)
Format:    JSON (orjson-serialized PatientStateSnapshot)

Operationally, Redis can be flushed at any time without data loss — the next
read falls through to Postgres.
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional

import orjson
import redis.asyncio as aioredis

from graphrag.config.settings import settings
from memory.schemas.state import PatientStateSnapshot

logger = logging.getLogger(__name__)


# Default TTL — read from settings if RuntimeConfig has the field; else 300 s.
_DEFAULT_TTL_SEC = getattr(settings, "MEMORY_CACHE_TTL_SEC", 300)


def _state_key(patient_id: uuid.UUID) -> str:
    return f"patient_state:{patient_id}"


class RedisStateCache:
    """Async Redis-backed cache for patient state snapshots."""

    def __init__(
        self,
        redis_url: str | None = None,
        ttl_seconds: int | None = None,
    ) -> None:
        self._url = redis_url or settings.REDIS_URL
        self._ttl = ttl_seconds or _DEFAULT_TTL_SEC
        self._client: aioredis.Redis | None = None

    async def _get_client(self) -> aioredis.Redis:
        if self._client is None:
            self._client = aioredis.from_url(
                self._url,
                encoding=None,            # we serialize ourselves with orjson
                decode_responses=False,
                socket_timeout=2.0,
                socket_connect_timeout=2.0,
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def get(self, patient_id: uuid.UUID) -> Optional[PatientStateSnapshot]:
        """Return cached snapshot or None on miss / error (degrades gracefully)."""
        try:
            client = await self._get_client()
            raw = await client.get(_state_key(patient_id))
        except Exception as exc:
            logger.warning("Redis GET failed for patient %s: %s", patient_id, exc)
            return None
        if raw is None:
            return None
        try:
            payload = orjson.loads(raw)
            return PatientStateSnapshot.model_validate(payload)
        except Exception as exc:
            logger.warning("Cached snapshot for %s was corrupt; dropping: %s", patient_id, exc)
            await self.invalidate(patient_id)
            return None

    async def set(self, snapshot: PatientStateSnapshot) -> None:
        """Write snapshot to cache. Silent on failure — cache is best-effort."""
        try:
            client = await self._get_client()
            payload = orjson.dumps(snapshot.model_dump(mode="json"))
            await client.setex(_state_key(snapshot.patient_id), self._ttl, payload)
        except Exception as exc:
            logger.warning("Redis SET failed for patient %s: %s", snapshot.patient_id, exc)

    async def invalidate(self, patient_id: uuid.UUID) -> None:
        """Drop the cached snapshot. Called on every write to clinical_fact."""
        try:
            client = await self._get_client()
            await client.delete(_state_key(patient_id))
        except Exception as exc:
            logger.warning("Redis DEL failed for patient %s: %s", patient_id, exc)
