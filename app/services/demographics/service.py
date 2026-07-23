"""
DemographicsService — load + build the AI-safe demographic context, fail-open.

This is the boundary the orchestrator talks to. It guarantees that NOTHING here
can break a chat turn:
  * disabled, no user_id, or no repository  -> None
  * user not found / no demographic fields   -> None
  * any Mongo error or build error           -> logged safely, None

The service returns a ``DemographicContextV1`` or ``None``. Relevance selection
(what to actually inject) happens later, in ``relevance.py``.
"""

from __future__ import annotations

import logging
from typing import Optional

from app.services.demographics.repository import DemographicsRepository
from app.services.demographics.types import (
    DemographicContextV1,
    build_demographic_context_v1,
)

logger = logging.getLogger(__name__)


def _safe_uid(user_id: Optional[str]) -> str:
    """A non-identifying hint for logs (never log the full id)."""
    if not user_id:
        return "<none>"
    return f"{user_id[:4]}…({len(user_id)} chars)"


class DemographicsService:
    """Fail-open loader for AI-safe patient demographics."""

    def __init__(
        self,
        repository: Optional[DemographicsRepository],
        *,
        enabled: bool = True,
        client: object | None = None,
    ) -> None:
        self._repo = repository
        self._enabled = enabled and repository is not None
        # Held only so the container can close the Mongo connection on shutdown.
        self._client = client

    def close(self) -> None:
        """Close the underlying Mongo client, if any. Safe to call always."""
        client = self._client
        if client is not None and hasattr(client, "close"):
            try:
                client.close()
            except Exception as exc:  # noqa: BLE001
                logger.debug("Mongo client close failed: %s", exc)

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def load(self, user_id: Optional[str]) -> Optional[DemographicContextV1]:
        """Return AI-safe demographics for ``user_id``, or None (always safe)."""
        if not self._enabled or not user_id:
            return None
        try:
            doc = await self._repo.fetch(user_id)  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001 — demographics must never break chat
            logger.warning("Demographics fetch failed (user_id=%s): %s", _safe_uid(user_id), exc)
            return None

        if not doc:
            logger.debug("No demographic record for user_id=%s", _safe_uid(user_id))
            return None

        try:
            ctx = build_demographic_context_v1(doc)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Demographics build failed (user_id=%s): %s", _safe_uid(user_id), exc)
            return None

        return None if ctx.is_empty() else ctx


def build_demographics_service(settings) -> DemographicsService:
    """
    Construct the demographics service from settings. Fails open at boot: if
    demographics are disabled, MONGO_URI is unset, or the client can't be
    created, a disabled service (always returns None) is returned instead of
    raising — the app boots and /chat works exactly as before.
    """
    if not getattr(settings, "DEMOGRAPHICS_ENABLED", False) or not getattr(settings, "MONGO_URI", None):
        logger.info("Demographics disabled (DEMOGRAPHICS_ENABLED off or MONGO_URI unset).")
        return DemographicsService(None, enabled=False)

    try:
        from pymongo import MongoClient

        client = MongoClient(
            settings.MONGO_URI,
            serverSelectionTimeoutMS=settings.MONGO_TIMEOUT_MS,
            connectTimeoutMS=settings.MONGO_TIMEOUT_MS,
            socketTimeoutMS=settings.MONGO_TIMEOUT_MS,
            appname="enervera-demographics-ro",
        )
        collection = client[settings.MONGO_DB][settings.MONGO_USERS_COLLECTION]
        repo = DemographicsRepository(collection)
        logger.info(
            "Demographics service active (db=%s collection=%s).",
            settings.MONGO_DB,
            settings.MONGO_USERS_COLLECTION,
        )
        return DemographicsService(repo, enabled=True, client=client)
    except Exception as exc:  # noqa: BLE001 — never block boot on demographics
        logger.warning("Demographics service init failed; continuing without it: %s", exc)
        return DemographicsService(None, enabled=False)


__all__ = ["DemographicsService", "build_demographics_service"]
