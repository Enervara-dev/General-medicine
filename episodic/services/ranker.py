"""
Ranker — pure math. Combines five factors into one score.

    score = (
        w_sim   * similarity_score          # cosine sim from Pinecone, in [0,1]
      + w_rec   * recency_score             # decay_score, in [0,1]
      + w_prio  * priority_weight           # 1.0 / 0.75 / 0.5 / 0.25
      + w_conf  * confidence                # episode confidence, in [0,1]
      + w_recur * recurrence_boost          # count-based, in [0,1]
    )

Weights are read from EpisodicConfig and sum to 1.0 by convention.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Iterable

from episodic.config import EpisodicConfig
from episodic.schemas.episode import ClinicalPriority, Episode
from episodic.schemas.retrieval import RankedEpisode
from episodic.services.decay import compute_decay_score


_PRIORITY_WEIGHTS: dict[ClinicalPriority, float] = {
    ClinicalPriority.CRITICAL: 1.00,
    ClinicalPriority.HIGH:     0.80,
    ClinicalPriority.MEDIUM:   0.55,
    ClinicalPriority.LOW:      0.30,
}


def _recurrence_boost(
    ep: Episode,
    *,
    counts_by_entity: Counter[str],
    max_count: int,
) -> float:
    """
    Boost episodes whose canonical entity appears multiple times in the result set.
    Returns [0, 1]: 0 if entity never repeats, 1 if it's the most frequent.
    """
    if max_count <= 1:
        return 0.0
    # Use the first symptom or first condition as the canonical entity key.
    key = (
        (ep.entities.symptoms[0] if ep.entities.symptoms else None)
        or (ep.entities.conditions[0] if ep.entities.conditions else None)
        or (ep.entities.medications[0] if ep.entities.medications else None)
    )
    if not key:
        return 0.0
    count = counts_by_entity.get(key.lower(), 0)
    return min(1.0, (count - 1) / (max_count - 1))


def rank_episodes(
    candidates: Iterable[tuple[Episode, float]],
    *,
    now: datetime | None = None,
    return_k: int | None = None,
) -> list[RankedEpisode]:
    """
    Re-rank Pinecone results by the composite score.

    `candidates` is the raw (episode, similarity) list from the repository.
    """
    candidates = list(candidates)
    if not candidates:
        return []

    # Pre-compute entity frequencies for the recurrence boost.
    canonical: list[str] = []
    for ep, _ in candidates:
        key = (
            (ep.entities.symptoms[0] if ep.entities.symptoms else None)
            or (ep.entities.conditions[0] if ep.entities.conditions else None)
            or (ep.entities.medications[0] if ep.entities.medications else None)
        )
        if key:
            canonical.append(key.lower())
    counts = Counter(canonical)
    max_count = max(counts.values()) if counts else 0

    w_sim = EpisodicConfig.RANK_W_SIMILARITY
    w_rec = EpisodicConfig.RANK_W_RECENCY
    w_pri = EpisodicConfig.RANK_W_PRIORITY
    w_con = EpisodicConfig.RANK_W_CONFIDENCE
    w_rcr = EpisodicConfig.RANK_W_RECURRENCE

    ranked: list[RankedEpisode] = []
    for ep, sim in candidates:
        recency = compute_decay_score(ep, now=now)
        priority = _PRIORITY_WEIGHTS.get(ep.clinical_priority, 0.5)
        confidence = float(ep.confidence)
        recurrence = _recurrence_boost(
            ep, counts_by_entity=counts, max_count=max_count
        )

        sim_norm = max(0.0, min(1.0, float(sim)))
        score = (
            w_sim * sim_norm
            + w_rec * recency
            + w_pri * priority
            + w_con * confidence
            + w_rcr * recurrence
        )
        ranked.append(
            RankedEpisode(
                episode=ep,
                score=score,
                factors={
                    "similarity": sim_norm,
                    "recency": recency,
                    "priority": priority,
                    "confidence": confidence,
                    "recurrence": recurrence,
                },
            )
        )

    ranked.sort(key=lambda r: r.score, reverse=True)
    if return_k is not None:
        ranked = ranked[:return_k]
    return ranked
