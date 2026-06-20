"""
RetrieverService — composes embedding → Pinecone query → composite rerank.

Optionally takes a `RetrievalEventLogger` that records every (request,
ranked_results) pair to an append-only JSONL for offline ranking eval.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, TYPE_CHECKING

from episodic.config import EpisodicConfig
from episodic.schemas.retrieval import RankedEpisode, RetrievalRequest
from episodic.services.ranker import rank_episodes
from episodic.services.storage import EpisodicRepository
from episodic.utils.embeddings import EmbeddingClient

if TYPE_CHECKING:
    from episodic.eval.logger import RetrievalEventLogger

logger = logging.getLogger(__name__)


class RetrieverService:
    def __init__(
        self,
        *,
        repository: EpisodicRepository,
        embedder: EmbeddingClient | None = None,
        event_logger: "RetrievalEventLogger | None" = None,
    ) -> None:
        self._repo = repository
        self._embedder = embedder or EmbeddingClient()
        self._event_logger = event_logger

    async def retrieve(self, req: RetrievalRequest) -> list[RankedEpisode]:
        top_k = req.top_k or EpisodicConfig.DEFAULT_TOP_K
        return_k = req.return_k or EpisodicConfig.DEFAULT_RETURN_K

        query_vector = await self._embedder.embed_query(req.query_text)
        metadata_filter = _build_metadata_filter(req)

        raw = await self._repo.query(
            user_id=req.user_id,
            query_vector=query_vector,
            top_k=top_k,
            metadata_filter=metadata_filter,
        )

        ranked = rank_episodes(raw, return_k=return_k)

        if self._event_logger is not None:
            # Logger swallows its own errors — never blocks the response.
            await self._event_logger.log(request=req, ranked=ranked)

        return ranked


def _build_metadata_filter(req: RetrievalRequest) -> dict[str, Any] | None:
    """
    Translate the RetrievalRequest filters into Pinecone's filter dialect.
    Returns None if there are no filters to apply (avoids unnecessary scan-narrowing).
    """
    f: dict[str, Any] = {}

    if req.categories:
        f["category"] = {"$in": list(req.categories)}

    ts_filter: dict[str, int] = {}
    if isinstance(req.since, datetime):
        ts_filter["$gte"] = int(req.since.timestamp())
    if isinstance(req.until, datetime):
        ts_filter["$lte"] = int(req.until.timestamp())
    if ts_filter:
        f["timestamp"] = ts_filter

    return f or None
