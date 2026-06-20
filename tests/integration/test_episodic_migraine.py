"""
migraine_test persona — assertions against the live episodic memory layer.

Run:
    python -m episodic.scripts.seed --persona migraine_test
    pytest -m episodic_integration tests/integration/test_episodic_migraine.py -v
"""

from __future__ import annotations

import pytest

from episodic.schemas.contradiction import ContradictionSeverity
from episodic.schemas.episode import EpisodeCategory
from episodic.schemas.retrieval import RetrievalRequest


pytestmark = pytest.mark.episodic_integration


USER_ID = "migraine_test"


@pytest.mark.asyncio
async def test_migraine_compression_collapses_recurring_episodes(episodic_container):
    """5 migraine episodes should compress to one cluster with members≥3."""
    block = await episodic_container.context_pipeline.build(
        RetrievalRequest(
            user_id=USER_ID,
            query_text="my migraine pattern over the last two months",
            top_k=20,
            return_k=10,
        )
    )
    assert block.compressed, "expected ≥1 compressed cluster for recurring migraines"
    migraine_clusters = [c for c in block.compressed if c.category == "symptom"]
    assert migraine_clusters, "expected a symptom-category cluster"
    biggest = max(migraine_clusters, key=lambda c: len(c.member_ids))
    assert len(biggest.member_ids) >= 3, (
        f"expected biggest migraine cluster to have ≥3 members; got {len(biggest.member_ids)}"
    )


@pytest.mark.asyncio
async def test_clarifier_caps_at_one_question(episodic_container):
    """If the clarifier fires, it must emit at most ONE question. Hard contract."""
    resp = await episodic_container.clarifier.evaluate(
        user_id=USER_ID,
        utterance="I had a bad headache yesterday.",
    )
    assert len(resp.questions) <= 1, (
        f"clarifier must emit ≤1 question; got {len(resp.questions)}"
    )
    if resp.needs_clarification:
        assert len(resp.questions) == 1, (
            "needs_clarification=True must come with exactly 1 question"
        )


@pytest.mark.asyncio
async def test_contradiction_flags_stopped_propranolol_as_current(episodic_container):
    """Patient claims to still take propranolol vs the 'discontinued' episode → contradiction."""
    ranked = await episodic_container.retriever.retrieve(
        RetrievalRequest(
            user_id=USER_ID,
            query_text="propranolol migraine prophylaxis",
            top_k=10,
            return_k=10,
        )
    )
    report = await episodic_container.contradiction.detect(
        user_id=USER_ID,
        new_claim="I'm currently taking propranolol daily for my migraines.",
        prior_episodes=[r.episode for r in ranked],
    )
    assert report.has_contradictions, "expected has_contradictions=True"
    assert any(
        c.severity in (ContradictionSeverity.WARNING, ContradictionSeverity.CRITICAL)
        for c in report.contradictions
    ), "expected severity ≥ warning"


@pytest.mark.asyncio
async def test_propranolol_episodes_rank_for_medication_query(episodic_container):
    """For 'what medications have I tried for migraines?', propranolol must appear in top 5."""
    ranked = await episodic_container.retriever.retrieve(
        RetrievalRequest(
            user_id=USER_ID,
            query_text="what medications have I tried for migraines?",
            return_k=5,
        )
    )
    top_text = " ".join(
        r.episode.summary.lower() + " " + " ".join(r.episode.entities.medications).lower()
        for r in ranked
    )
    assert "propranolol" in top_text, (
        f"expected propranolol referenced in top 5; got summaries: "
        f"{[r.episode.summary for r in ranked]}"
    )
