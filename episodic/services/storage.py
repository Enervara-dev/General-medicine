"""
EpisodicRepository — protocol + Pinecone implementation.

Pinecone metadata constraints are flat (str/num/bool/list[str]). We store
filterable fields as metadata AND embed the full Episode as a JSON-encoded
`payload` field for lossless rehydration. A future graph-backed repository
can implement the same Protocol without changing service callers.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Iterable, Protocol

from pinecone import Pinecone, ServerlessSpec

from episodic.config import EpisodicConfig
from episodic.schemas.episode import Episode
from episodic.utils.embeddings import EmbeddingClient
from episodic.utils.retry import async_retry

logger = logging.getLogger(__name__)


class EpisodicRepository(Protocol):
    async def upsert(self, episode: Episode) -> None: ...
    async def upsert_batch(self, episodes: Iterable[Episode]) -> None: ...
    async def query(
        self,
        *,
        user_id: str,
        query_vector: list[float],
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[tuple[Episode, float]]: ...
    async def fetch(
        self,
        *,
        user_id: str,
        episode_ids: list[uuid.UUID],
    ) -> list[Episode]: ...
    async def delete(
        self,
        *,
        user_id: str,
        episode_ids: list[uuid.UUID],
    ) -> None: ...


class PineconeEpisodicRepository:
    """Pinecone-backed episodic storage. Namespace per user_id."""

    def __init__(
        self,
        *,
        pc: Pinecone | None = None,
        embedder: EmbeddingClient | None = None,
    ) -> None:
        if not EpisodicConfig.PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY is missing.")
        self._pc = pc or Pinecone(api_key=EpisodicConfig.PINECONE_API_KEY)
        self._embedder = embedder or EmbeddingClient(self._pc)
        self._index_name = EpisodicConfig.PINECONE_INDEX_NAME
        self._index = None  # lazy

    # ------------------------------------------------------------------
    # Index bootstrap
    # ------------------------------------------------------------------

    async def ensure_index(self) -> None:
        """Create the index if it does not exist. Idempotent."""
        existing = await asyncio.to_thread(self._pc.list_indexes)
        names = {i.name for i in existing}
        if self._index_name not in names:
            logger.info("Creating Pinecone episodic index: %s", self._index_name)
            await asyncio.to_thread(
                self._pc.create_index,
                name=self._index_name,
                dimension=EpisodicConfig.PINECONE_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=EpisodicConfig.PINECONE_CLOUD,
                    region=EpisodicConfig.PINECONE_REGION,
                ),
            )

    def _get_index(self):
        if self._index is None:
            self._index = self._pc.Index(self._index_name)
        return self._index

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------

    @async_retry(attempts=3, base_delay_s=0.5)
    async def upsert(self, episode: Episode) -> None:
        await self.upsert_batch([episode])

    @async_retry(attempts=3, base_delay_s=0.5)
    async def upsert_batch(self, episodes: Iterable[Episode]) -> None:
        episodes = [e for e in episodes if e.store_memory]
        if not episodes:
            return

        vectors = await self._embedder.embed_passages(e.embedding_text for e in episodes)

        # Group by user_id for namespace-correct upserts.
        by_ns: dict[str, list[dict]] = {}
        for episode, vec in zip(episodes, vectors):
            by_ns.setdefault(episode.user_id, []).append(
                {
                    "id": str(episode.episode_id),
                    "values": vec,
                    "metadata": _episode_to_metadata(episode),
                }
            )

        index = self._get_index()
        for namespace, items in by_ns.items():
            # Batch in chunks of 100 — Pinecone hard limit.
            for start in range(0, len(items), 100):
                chunk = items[start : start + 100]
                await asyncio.to_thread(
                    index.upsert, vectors=chunk, namespace=namespace
                )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    @async_retry(attempts=3, base_delay_s=0.5)
    async def query(
        self,
        *,
        user_id: str,
        query_vector: list[float],
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[tuple[Episode, float]]:
        index = self._get_index()
        result = await asyncio.to_thread(
            index.query,
            vector=query_vector,
            top_k=top_k,
            namespace=user_id,
            include_metadata=True,
            filter=metadata_filter or None,
        )
        out: list[tuple[Episode, float]] = []
        for match in result.matches:
            ep = _metadata_to_episode(match.metadata)
            if ep is not None:
                out.append((ep, float(match.score)))
        return out

    @async_retry(attempts=3, base_delay_s=0.5)
    async def fetch(
        self,
        *,
        user_id: str,
        episode_ids: list[uuid.UUID],
    ) -> list[Episode]:
        if not episode_ids:
            return []
        index = self._get_index()
        result = await asyncio.to_thread(
            index.fetch,
            ids=[str(eid) for eid in episode_ids],
            namespace=user_id,
        )
        episodes: list[Episode] = []
        for vec in (result.vectors or {}).values():
            ep = _metadata_to_episode(getattr(vec, "metadata", None) or {})
            if ep is not None:
                episodes.append(ep)
        return episodes

    @async_retry(attempts=3, base_delay_s=0.5)
    async def delete(
        self,
        *,
        user_id: str,
        episode_ids: list[uuid.UUID],
    ) -> None:
        if not episode_ids:
            return
        index = self._get_index()
        await asyncio.to_thread(
            index.delete,
            ids=[str(eid) for eid in episode_ids],
            namespace=user_id,
        )


# ---------------------------------------------------------------------------
# Pinecone metadata <-> Episode marshalling
# ---------------------------------------------------------------------------


def _episode_to_metadata(ep: Episode) -> dict[str, Any]:
    """
    Flatten an Episode into Pinecone-compatible metadata.

    Filterable fields are emitted as primitives or list[str]. The full
    Episode is serialized as a JSON string in `payload` for lossless rehydration.
    Pinecone metadata strings are capped at 40KB — well above what we need.
    """
    md: dict[str, Any] = {
        "episode_id": str(ep.episode_id),
        "user_id": ep.user_id,
        "timestamp": int(ep.timestamp.timestamp()),  # epoch seconds for range filters
        "category": ep.category.value,
        "severity": ep.severity.value,
        "clinical_priority": ep.clinical_priority.value,
        "confidence": float(ep.confidence),
        "source": ep.source,
        "summary": ep.summary[:1000],  # safety cap
        # entity lists — Pinecone metadata supports list[str]
        "symptoms": list(ep.entities.symptoms),
        "conditions": list(ep.entities.conditions),
        "medications": list(ep.entities.medications),
        "labs": list(ep.entities.labs),
        "body_parts": list(ep.entities.body_parts),
    }
    # Optional temporal fields — omit empty strings to keep metadata lean.
    for field in ("duration", "onset", "frequency", "progression"):
        v = getattr(ep.temporal_data, field)
        if v:
            md[field] = v

    # Full payload for lossless rehydration (mode='json' handles datetime + UUID).
    md["payload"] = json.dumps(ep.model_dump(mode="json"))
    return md


def _metadata_to_episode(metadata: dict[str, Any] | None) -> Episode | None:
    if not metadata:
        return None
    payload = metadata.get("payload")
    if payload:
        try:
            return Episode.model_validate_json(payload)
        except Exception as exc:
            logger.warning("Failed to rehydrate Episode from payload: %s", exc)
    # Fallback: try to reconstruct from flat metadata only (lossy).
    try:
        ts = metadata.get("timestamp")
        if isinstance(ts, (int, float)):
            iso = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        else:
            iso = ts
        return Episode.model_validate(
            {
                "episode_id": metadata.get("episode_id", str(uuid.uuid4())),
                "user_id": metadata.get("user_id", "unknown"),
                "timestamp": iso or datetime.now(tz=timezone.utc).isoformat(),
                "summary": metadata.get("summary", ""),
                "category": metadata.get("category", "symptom"),
                "severity": metadata.get("severity", "unknown"),
                "clinical_priority": metadata.get("clinical_priority", "medium"),
                "confidence": metadata.get("confidence", 0.7),
                "source": metadata.get("source", "user_self_report"),
                "embedding_text": metadata.get("summary", ""),
                "entities": {
                    "symptoms": metadata.get("symptoms", []),
                    "conditions": metadata.get("conditions", []),
                    "medications": metadata.get("medications", []),
                    "labs": metadata.get("labs", []),
                    "body_parts": metadata.get("body_parts", []),
                },
                "temporal_data": {
                    k: metadata.get(k, "")
                    for k in ("duration", "onset", "frequency", "progression")
                },
            }
        )
    except Exception as exc:
        logger.warning("Failed to reconstruct Episode from flat metadata: %s", exc)
        return None
