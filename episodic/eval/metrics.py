"""
Pure ranking metrics. No I/O, no async, no dependencies beyond stdlib.

Each function takes a list of retrieved episode IDs (ordered, rank 1 first)
and a set of relevant episode IDs (the ground truth). Returns a float in
[0, 1] except for MRR which returns [0, 1] too (reciprocal of first hit).
"""

from __future__ import annotations

import math
from typing import Iterable


def precision_at_k(retrieved: list[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of the top-k retrieved items that are in the relevant set."""
    if k <= 0:
        return 0.0
    relevant_set = set(relevant)
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for r in top_k if r in relevant_set)
    return hits / min(k, len(top_k))


def recall_at_k(retrieved: list[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of relevant items that appear in the top-k retrieved."""
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    top_k = set(retrieved[:k])
    return len(top_k & relevant_set) / len(relevant_set)


def reciprocal_rank(retrieved: list[str], relevant: Iterable[str]) -> float:
    """1 / rank of the first relevant item. 0 if none in the result list."""
    relevant_set = set(relevant)
    for i, r in enumerate(retrieved, start=1):
        if r in relevant_set:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: Iterable[str], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain at k with binary relevance.

    DCG = sum_i (rel_i / log2(i+1)) for i in 1..k
    IDCG = best possible DCG given the number of relevant items capped at k
    nDCG = DCG / IDCG (1.0 = perfect ordering, 0.0 = no hits)
    """
    if k <= 0:
        return 0.0
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0

    dcg = 0.0
    for i, r in enumerate(retrieved[:k], start=1):
        if r in relevant_set:
            dcg += 1.0 / math.log2(i + 1)

    ideal_hits = min(k, len(relevant_set))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def aggregate(events: list[dict], k: int = 5) -> dict[str, float]:
    """
    Compute mean precision@k / recall@k / MRR / nDCG@k over a list of
    {retrieved: [...], relevant: [...]} dicts.

    Skips events with empty `relevant` lists (they're unlabeled).
    """
    p_at_k: list[float] = []
    r_at_k: list[float] = []
    mrr: list[float] = []
    ndcg: list[float] = []
    for ev in events:
        relevant = ev.get("relevant") or []
        if not relevant:
            continue
        retrieved = ev.get("retrieved") or []
        p_at_k.append(precision_at_k(retrieved, relevant, k))
        r_at_k.append(recall_at_k(retrieved, relevant, k))
        mrr.append(reciprocal_rank(retrieved, relevant))
        ndcg.append(ndcg_at_k(retrieved, relevant, k))

    n = len(p_at_k)
    if n == 0:
        return {"labeled_events": 0}
    return {
        "labeled_events": n,
        f"precision@{k}": sum(p_at_k) / n,
        f"recall@{k}": sum(r_at_k) / n,
        "mrr": sum(mrr) / n,
        f"ndcg@{k}": sum(ndcg) / n,
    }
