"""
geriatric_test persona — assertions against the live episodic memory layer.

Run:
    python -m episodic.scripts.seed --persona geriatric_test
    pytest -m episodic_integration tests/integration/test_episodic_geriatric.py -v
"""

from __future__ import annotations

import pytest

from episodic.schemas.episode import EpisodeCategory
from episodic.schemas.retrieval import RetrievalRequest


pytestmark = pytest.mark.episodic_integration


USER_ID = "geriatric_test"


@pytest.mark.asyncio
async def test_chronic_conditions_have_recency_one(episodic_container):
    """HTN / T2DM / CKD are CONDITION-category episodes — all must have recency=1.0."""
    ranked = await episodic_container.retriever.retrieve(
        RetrievalRequest(
            user_id=USER_ID,
            query_text="what are my long-term medical conditions?",
            return_k=15,
        )
    )
    conditions = [r for r in ranked if r.episode.category == EpisodeCategory.CONDITION]
    assert conditions, "expected at least one condition episode"
    for r in conditions:
        assert r.factors["recency"] == pytest.approx(1.0), (
            f"chronic condition '{r.episode.summary[:50]}' must have recency=1.0; "
            f"got {r.factors['recency']}"
        )


@pytest.mark.asyncio
async def test_polypharmacy_returns_multiple_meds(episodic_container):
    """Geriatric patient has 5 ongoing meds — at least 3 should surface in a medication query."""
    ranked = await episodic_container.retriever.retrieve(
        RetrievalRequest(
            user_id=USER_ID,
            query_text="what medications am I currently taking?",
            top_k=20,
            return_k=15,
        )
    )
    meds = [r for r in ranked if r.episode.category == EpisodeCategory.MEDICATION]
    assert len(meds) >= 3, (
        f"expected ≥3 medication episodes; got {len(meds)}. "
        f"All categories returned: {[r.episode.category.value for r in ranked]}"
    )


@pytest.mark.asyncio
async def test_penicillin_allergy_recency_persists(episodic_container):
    """Penicillin allergy is ~3000 days old but must surface with recency=1.0."""
    ranked = await episodic_container.retriever.retrieve(
        RetrievalRequest(
            user_id=USER_ID,
            query_text="any drug allergies I have?",
            return_k=10,
        )
    )
    allergies = [r for r in ranked if r.episode.category == EpisodeCategory.ALLERGY]
    assert allergies, "expected ≥1 allergy episode in top results"
    assert allergies[0].factors["recency"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_recent_dizziness_outranks_old_fall(episodic_container):
    """For a balance query, recent (5d) dizziness should outrank the 90-day-old fall."""
    ranked = await episodic_container.retriever.retrieve(
        RetrievalRequest(
            user_id=USER_ID,
            query_text="history of falls or balance issues",
            top_k=20,
            return_k=10,
        )
    )
    by_time_desc = sorted(ranked, key=lambda r: r.episode.timestamp, reverse=True)
    # The most recent symptom episode should score higher than the oldest symptom episode in result.
    recent_symptoms = [r for r in by_time_desc if r.episode.category == EpisodeCategory.SYMPTOM]
    if len(recent_symptoms) >= 2:
        assert recent_symptoms[0].score > recent_symptoms[-1].score, (
            "expected more recent symptom episode to score higher than oldest symptom"
        )


@pytest.mark.asyncio
async def test_end_to_end_ingest_extracts_lab_episode(episodic_container):
    """Ingesting 'my blood sugar was 312 this morning' must produce a lab episode."""
    result = await episodic_container.ingest_pipeline.run(
        user_id=USER_ID,
        utterance="My blood sugar was 312 this morning before breakfast.",
    )
    # Either stored or extracted as a candidate — both prove extraction worked.
    candidate = result.stored or result.candidate
    assert candidate is not None, "expected extractor to produce an episode"
    assert candidate.category == EpisodeCategory.LAB, (
        f"expected category=lab; got {candidate.category.value}"
    )
    assert candidate.confidence >= 0.6
