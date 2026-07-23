"""
Read-only MongoDB access for patient demographics.

The repository is the ONLY place that touches Mongo. It:
  * reads from `enervara.users` (never writes),
  * projects to AI-safe fields ONLY — sensitive fields (email, phone, password,
    firebaseUID, token hashes, _id) never leave the database, and
  * resolves the request `user_id` to a document by `_id` (ObjectId) first,
    falling back to `firebaseUID` — covering both id conventions a frontend
    might send.

The sync pymongo call is wrapped in ``asyncio.to_thread`` so it never blocks the
event loop, matching the Pinecone/Neo4j pattern used elsewhere.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from app.services.demographics.types import AI_SAFE_MONGO_FIELDS

logger = logging.getLogger(__name__)


class DemographicsRepository:
    """Fetches an AI-safe projection of one user document. Read-only."""

    def __init__(self, collection: Any) -> None:
        # `collection` is a pymongo Collection (enervara.users). Injected so the
        # repo is unit-testable with a fake collection (no network).
        self._col = collection
        # Return ONLY AI-safe fields; explicitly drop _id. This is the hard
        # guarantee that no sensitive field is ever read out of Mongo.
        self._projection: dict[str, int] = {f: 1 for f in AI_SAFE_MONGO_FIELDS}
        self._projection["_id"] = 0

    def _find_sync(self, user_id: str) -> Optional[dict[str, Any]]:
        # Primary: the Mongo _id (24-char hex ObjectId).
        try:
            from bson import ObjectId
            from bson.errors import InvalidId

            try:
                oid = ObjectId(user_id)
            except (InvalidId, TypeError, ValueError):
                oid = None
            if oid is not None:
                doc = self._col.find_one({"_id": oid}, self._projection)
                if doc:
                    return doc
        except ImportError:  # bson always ships with pymongo; defensive only
            pass

        # Fallback: firebaseUID (what an auth-backed frontend often uses).
        return self._col.find_one({"firebaseUID": user_id}, self._projection)

    async def fetch(self, user_id: str) -> Optional[dict[str, Any]]:
        """Return the AI-safe projected doc for ``user_id``, or None if absent."""
        return await asyncio.to_thread(self._find_sync, user_id)


__all__ = ["DemographicsRepository"]
