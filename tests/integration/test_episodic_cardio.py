"""
cardio_test persona — assertions against the live episodic memory layer.

Run:
    python -m episodic.scripts.seed --persona cardio_test
    pytest -m episodic_integration tests/integration/test_episodic_cardio.py -v
"""

from __future__ import annotations

import pytest

from episodic.schemas.contradiction import ContradictionSeverity
from episodic.schemas.episode import ClinicalPriority, EpisodeCategory
from episodic.schemas.retrieval import RetrievalRequest


pytestmark = pytest.mark.episodic_integration


USER_ID = "cardio_test"


@pytest.mark.asyncio
async def test_recall_returns_heart_related_top(episodic_container):
    """The top result for a heart-related query should be in a clinical category, not a lifestyle/lab note."""
    ranked = await episodic_container.retriever.retrieve(
        RetrievalRequest(
            user_id=USER_ID,
            query_text="Have I had any heart-related issues recently?",
        )
    )
    assert len(ranked) >= 3, "expected at least 3 ranked episodes"

    clinical_categories = {
        EpisodeCategory.SYMPTOM,
        EpisodeCategory.CONSULTATION,
        EpisodeCategory.MEDICATION,
        EpisodeCategory.ALLERGY,
        EpisodeCategory.CONDITION,
    }
    top_categories = {r.episode.category for r in ranked[:3]}
    assert top_categories & clinical_categories, (
        "expected at least one clinical-category episode in top 3, got "
        f"{[r.episode.category.value for r in ranked[:3]]}"
    )


@pytest.mark.asyncio
async def test_decay_orders_chest_pain_by_recency(episodic_container):
    """Within results matching 'chest pain on exertion', recency factor should monotone-decrease with episode age."""
    ranked = await episodic_container.retriever.retrieve(
        RetrievalRequest(
            user_id=USER_ID,
            query_text="chest pain on exertion",
            top_k=20,
            return_k=12,
        )
    )
    chest_pain_eps = [
        r for r in ranked
        if r.episode.category == EpisodeCategory.SYMPTOM
        and "chest" in " ".join(r.episode.entities.body_parts).lower()
    ]
    assert len(chest_pain_eps) >= 3, (
        f"expected ≥3 chest-pain episodes in result, got {len(chest_pain_eps)}"
    )

    # Sort by episode timestamp asc → recency factor must be non-increasing as we go further back.
    by_time = sorted(chest_pain_eps, key=lambda r: r.episode.timestamp, reverse=True)
    recencies = [r.factors["recency"] for r in by_time]
    for older, newer in zip(recencies[1:], recencies[:-1]):
        assert older <= newer + 1e-6, (
            f"recency should monotone-decrease with age; got {recencies}"
        )


@pytest.mark.asyncio
async def test_aspirin_allergy_recency_persists(episodic_container):
    """Aspirin allergy is ~800 days old but should still surface with recency=1.0 (chronic)."""
    ranked = await episodic_container.retriever.retrieve(
        RetrievalRequest(
            user_id=USER_ID,
            query_text="what am I allergic to?",
            return_k=10,
        )
    )
    allergies = [r for r in ranked if r.episode.category == EpisodeCategory.ALLERGY]
    assert allergies, "expected at least one allergy episode in results"
    assert allergies[0].factors["recency"] == pytest.approx(1.0), (
        f"expected aspirin allergy recency=1.0, got {allergies[0].factors['recency']}"
    )


@pytest.mark.asyncio
async def test_contradiction_flags_denied_cardiac_history(episodic_container):
    """Claim 'I have never had any cardiac symptoms' must flag multiple critical contradictions."""
    ranked = await episodic_container.retriever.retrieve(
        RetrievalRequest(
            user_id=USER_ID,
            query_text="cardiac symptoms or chest pain history",
            top_k=10,
            return_k=10,
        )
    )
    report = await episodic_container.contradiction.detect(
        user_id=USER_ID,
        new_claim="I have never had any cardiac symptoms or chest pain in my life.",
        prior_episodes=[r.episode for r in ranked],
    )
    assert report.has_contradictions, "expected has_contradictions=True"
    assert report.triggers_clarification, "expected triggers_clarification=True"
    assert report.confidence_penalty > 0.0, "expected non-zero confidence_penalty"
    critical_or_warning = [
        c for c in report.contradictions
        if c.severity in (ContradictionSeverity.WARNING, ContradictionSeverity.CRITICAL)
    ]
    assert critical_or_warning, (
        "expected at least one contradiction at severity ≥ warning; got "
        f"{[c.severity.value for c in report.contradictions]}"
    )


@pytest.mark.asyncio
async def test_compression_clusters_chest_pain_history(episodic_container):
    """After the cluster_key fix, chest-pain variants should collapse into one cluster."""
    block = await episodic_container.context_pipeline.build(
        RetrievalRequest(
            user_id=USER_ID,
            query_text="tell me about my chest pain history",
            top_k=20,
            return_k=10,
        )
    )
    assert block.compressed, (
        "expected at least one CompressedEpisode cluster; got 0. "
        "Check episodic/services/compression.py cluster_key logic."
    )
    biggest = max(block.compressed, key=lambda c: len(c.member_ids))
    assert len(biggest.member_ids) >= 3, (
        f"expected biggest cluster to have ≥3 members; got {len(biggest.member_ids)}"
    )
    assert biggest.summary.strip(), "compressed cluster must have a non-empty summary"
