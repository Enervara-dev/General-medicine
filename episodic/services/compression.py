"""
CompressionService — collapses redundant episode clusters into one prose summary.

Trigger: any cluster with >=3 episodes about the same canonical entity is
compressed via Gemini. Smaller clusters pass through untouched.

Clustering key is intentionally simple (category + lowercased first
entity-name) because we run it on small result sets (≤20 episodes). For a
larger retrieval window, swap in a proper entity-resolution step.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict

from graphrag.llm.gemini_client import generate_text_async

from episodic.config import EpisodicConfig
from episodic.prompts.compression import COMPRESSION_SYSTEM_PROMPT
from episodic.schemas.episode import Episode, EpisodeCategory
from episodic.schemas.retrieval import CompressedEpisode, RankedEpisode

logger = logging.getLogger(__name__)


_MIN_CLUSTER_SIZE = 3


class CompressionService:
    def __init__(self, model: str | None = None) -> None:
        self._model = model or EpisodicConfig.COMPRESSION_MODEL

    async def compress(
        self,
        ranked: list[RankedEpisode],
    ) -> tuple[list[RankedEpisode], list[CompressedEpisode]]:
        """
        Returns (kept_individual_episodes, compressed_clusters).

        Episodes that belong to a compressed cluster are removed from the
        individual list and represented by the CompressedEpisode entry.
        """
        if len(ranked) < _MIN_CLUSTER_SIZE:
            return ranked, []

        clusters: dict[tuple[str, str], list[RankedEpisode]] = defaultdict(list)
        for r in ranked:
            key = _cluster_key(r.episode)
            if key is None:
                continue
            clusters[key].append(r)

        compressed: list[CompressedEpisode] = []
        absorbed_ids: set = set()

        for (category, entity), members in clusters.items():
            if len(members) < _MIN_CLUSTER_SIZE:
                continue
            try:
                summary = await self._summarize(members)
            except Exception as exc:
                logger.warning("Compression summary failed; keeping raw: %s", exc)
                continue
            if not summary:
                continue

            sorted_by_time = sorted(members, key=lambda r: r.episode.timestamp)
            representative = max(members, key=lambda r: r.score)
            peak_severity = _peak_severity([r.episode for r in members])

            compressed.append(
                CompressedEpisode(
                    representative_id=representative.episode.episode_id,
                    member_ids=[r.episode.episode_id for r in members],
                    category=category,
                    summary=summary,
                    first_seen=sorted_by_time[0].episode.timestamp,
                    last_seen=sorted_by_time[-1].episode.timestamp,
                    peak_severity=peak_severity,
                    score=representative.score,
                )
            )
            absorbed_ids.update(r.episode.episode_id for r in members)

        kept = [r for r in ranked if r.episode.episode_id not in absorbed_ids]
        return kept, compressed

    async def _summarize(self, members: list[RankedEpisode]) -> str:
        body = json.dumps(
            [
                {
                    "timestamp": m.episode.timestamp.isoformat(),
                    "summary": m.episode.summary,
                    "severity": m.episode.severity.value,
                    "duration": m.episode.temporal_data.duration,
                    "progression": m.episode.temporal_data.progression,
                    "frequency": m.episode.temporal_data.frequency,
                }
                for m in members
            ],
            default=str,
        )
        result = await generate_text_async(
            body,
            model=self._model,
            system_instruction=COMPRESSION_SYSTEM_PROMPT,
            temperature=0.2,
        )
        return (result or "").strip()


def _cluster_key(ep: Episode) -> tuple[str, str] | None:
    """
    Pick the canonical entity for clustering.

    For symptom episodes we prefer the body part (e.g. "chest") over the
    first symptom string, because human-authored symptom labels vary
    ("chest pain" vs "chest tightness" vs "chest pressure") and would
    otherwise fail to cluster. Conditions and medications use their first
    canonical entity which is already a stable name.
    """
    if ep.category == EpisodeCategory.SYMPTOM and ep.entities.body_parts:
        return (ep.category.value, ep.entities.body_parts[0].lower())
    entity = (
        (ep.entities.symptoms[0] if ep.entities.symptoms else None)
        or (ep.entities.conditions[0] if ep.entities.conditions else None)
        or (ep.entities.medications[0] if ep.entities.medications else None)
    )
    if not entity:
        return None
    return (ep.category.value, entity.lower())


def _peak_severity(episodes: list[Episode]) -> str | None:
    order = {"critical": 4, "severe": 3, "moderate": 2, "mild": 1, "unknown": 0}
    if not episodes:
        return None
    peak = max(episodes, key=lambda e: order.get(e.severity.value, 0))
    return peak.severity.value
