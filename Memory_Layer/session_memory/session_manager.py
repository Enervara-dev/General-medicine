"""
session_manager.py
──────────────────
Async Redis-backed session manager for the Enervera memory layer.
Includes a graceful in-memory fallback if Redis is unavailable.

Redis key schema:
  session:{session_id}   →  orjson-serialised SessionMemory
"""

from __future__ import annotations

import logging
import os
import json
from contextlib import asynccontextmanager
from typing import AsyncIterator
from uuid import uuid4

try:
    import orjson
except ImportError:
    orjson = None

try:
    import redis.asyncio as aioredis
    from redis.asyncio import Redis
except ImportError:
    aioredis = None
    Redis = object

from .models import SessionMemory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global In-Memory Fallback Store
# ---------------------------------------------------------------------------
# This persists for the lifetime of the python process if Redis is down.
_IN_MEMORY_STORE: dict[str, bytes] = {}
_WARNED_REDIS_UNAVAILABLE: bool = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REDIS_URL:       str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
SESSION_TTL_SEC: int = int(os.getenv("SESSION_TTL_SEC", str(60 * 60 * 2)))
KEY_PREFIX:      str = "session"

def _redis_key(session_id: str) -> str:
    return f"{KEY_PREFIX}:{session_id}"

# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def _serialize(session: SessionMemory) -> bytes:
    payload = session.model_dump(mode="json")
    if orjson is not None:
        return orjson.dumps(payload, option=orjson.OPT_UTC_Z)
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")

def _deserialize(raw: bytes) -> SessionMemory:
    data = orjson.loads(raw) if orjson is not None else json.loads(raw)
    return SessionMemory.model_validate(data)

# ---------------------------------------------------------------------------
# SessionManager
# ---------------------------------------------------------------------------

class SessionManager:
    def __init__(
        self,
        redis_url: str = REDIS_URL,
        ttl: int       = SESSION_TTL_SEC,
    ) -> None:
        self._redis_url = redis_url
        self._ttl       = ttl
        self._client: Redis | None = None
        self._use_fallback = False

    async def open(self) -> None:
        """Open Redis connection with automatic fallback to RAM if Redis is down."""
        global _WARNED_REDIS_UNAVAILABLE
        self._use_fallback = False
        
        if aioredis is None:
            if not _WARNED_REDIS_UNAVAILABLE:
                logger.warning("Redis package not installed. Using in-memory fallback.")
                _WARNED_REDIS_UNAVAILABLE = True
            self._use_fallback = True
            return

        try:
            self._client = aioredis.from_url(
                self._redis_url,
                encoding="utf-8",
                decode_responses=False,
                socket_connect_timeout=2, # Fast fail if redis is down
            )
            # Test connection
            await self._client.ping()
            logger.debug("SessionManager: Redis connection active.")
        except Exception as e:
            if not _WARNED_REDIS_UNAVAILABLE:
                logger.warning(f"Redis unavailable ({e}). Using in-memory fallback store.")
                _WARNED_REDIS_UNAVAILABLE = True
            self._use_fallback = True
            if self._client:
                await self._client.aclose()
                self._client = None

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "SessionManager":
        await self.open()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def create_session(self, session_id: str | None = None) -> SessionMemory:
        session = SessionMemory(session_id=session_id or uuid4().hex)
        await self.save_session(session)
        return session

    async def load_session(self, session_id: str) -> SessionMemory | None:
        key = _redis_key(session_id)
        
        if self._use_fallback:
            raw = _IN_MEMORY_STORE.get(key)
        else:
            try:
                raw = await self._client.get(key)
            except Exception:
                raw = _IN_MEMORY_STORE.get(key)

        if raw is None:
            return None

        return _deserialize(raw)

    async def save_session(self, session: SessionMemory) -> None:
        session._trim_turns()
        key  = _redis_key(session.session_id)
        data = _serialize(session)

        if self._use_fallback:
            _IN_MEMORY_STORE[key] = data
        else:
            try:
                await self._client.set(key, data, ex=self._ttl)
            except Exception:
                _IN_MEMORY_STORE[key] = data

    async def delete_session(self, session_id: str) -> bool:
        key = _redis_key(session_id)
        if self._use_fallback:
            return bool(_IN_MEMORY_STORE.pop(key, None))
        try:
            return bool(await self._client.delete(key))
        except Exception:
            return bool(_IN_MEMORY_STORE.pop(key, None))

@asynccontextmanager
async def session_context(redis_url: str = REDIS_URL, ttl: int = SESSION_TTL_SEC) -> AsyncIterator[SessionManager]:
    mgr = SessionManager(redis_url=redis_url, ttl=ttl)
    async with mgr:
        yield mgr
