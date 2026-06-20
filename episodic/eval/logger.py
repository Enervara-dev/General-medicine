"""
RetrievalEventLogger — append-only JSONL writer for ranking evaluation.

Each event captures everything needed to (a) reproduce the retrieval offline
and (b) compute ranking metrics against a labeled ground-truth set:

    - event_id, timestamp, user_id, query_text, top_k, return_k
    - the full ranked list with episode_id, score, factors, and short summary

The labels are stored in a separate file so re-runs of the same query don't
shadow previously-collected labels. Both files are JSONL — one record per
line — so they're trivially appendable, diffable, and grep-friendly.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from episodic.schemas.retrieval import RankedEpisode, RetrievalRequest

logger = logging.getLogger(__name__)


class RetrievalEventLogger:
    """Append-only JSONL logger of retrieval events."""

    def __init__(self, log_path: str | os.PathLike) -> None:
        self._path = Path(log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    async def log(
        self,
        *,
        request: RetrievalRequest,
        ranked: Iterable[RankedEpisode],
    ) -> str:
        """Write one event. Returns the event_id (UUID hex)."""
        event_id = uuid.uuid4().hex
        event = {
            "event_id": event_id,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "user_id": request.user_id,
            "query_text": request.query_text,
            "top_k": request.top_k,
            "return_k": request.return_k,
            "results": [
                {
                    "rank": i + 1,
                    "episode_id": str(r.episode.episode_id),
                    "score": float(r.score),
                    "factors": {k: float(v) for k, v in r.factors.items()},
                    "category": r.episode.category.value,
                    "summary": r.episode.summary[:200],
                }
                for i, r in enumerate(ranked)
            ],
        }
        try:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        except Exception as exc:
            # Eval logging must never break a real query — log and swallow.
            logger.warning("RetrievalEventLogger.log failed: %s", exc)
        return event_id

    @property
    def path(self) -> Path:
        return self._path
