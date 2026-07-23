"""Unit tests for the read-only demographics layer (no network)."""

from datetime import date, datetime

import pytest

from app.services.demographics import (
    DemographicContextV1,
    DemographicsRepository,
    DemographicsService,
    build_demographic_context_v1,
    derive_age,
    derive_bmi,
    render_demographic_block,
    select_relevant_fields,
)

TODAY = date(2026, 7, 23)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _FakeCollection:
    """Minimal stand-in for a pymongo Collection."""

    def __init__(self, docs, *, raise_exc=None):
        self._docs = docs
        self._raise = raise_exc
        self.last_projection = None

    def find_one(self, query, projection=None):
        if self._raise:
            raise self._raise
        self.last_projection = projection
        for d in self._docs:
            for k, v in query.items():
                if d.get(k) != v:
                    break
            else:
                # Apply projection (only the projected keys, minus _id:0).
                if projection:
                    keep = {k for k, on in projection.items() if on and k != "_id"}
                    return {k: val for k, val in d.items() if k in keep}
                return dict(d)
        return None


def _service(docs=None, *, raise_exc=None, enabled=True):
    repo = DemographicsRepository(_FakeCollection(docs or [], raise_exc=raise_exc))
    return DemographicsService(repo, enabled=enabled)


VALID_DOC = {
    "firebaseUID": "uid-123",
    "dateOfBirth": datetime(1990, 1, 1),
    "sex": "male",
    "heightCm": 175,
    "weightKg": 80,
    "state": "Telangana",
    "city": "Hyderabad",
    # sensitive fields that must NEVER surface:
    "email": "x@y.com",
    "phone": "999",
    "password": "secret",
}


# ---------------------------------------------------------------------------
# Derivation
# ---------------------------------------------------------------------------

def test_age_derivation():
    assert derive_age(datetime(1990, 1, 1), today=TODAY) == 36
    assert derive_age(datetime(2002, 4, 13), today=TODAY) == 24
    # Birthday not yet reached this year.
    assert derive_age(datetime(1990, 12, 31), today=TODAY) == 35
    assert derive_age("2000-06-15", today=TODAY) == 26
    assert derive_age(None, today=TODAY) is None
    assert derive_age("garbage", today=TODAY) is None


def test_bmi_derivation():
    assert derive_bmi(175, 80) == pytest.approx(26.1, abs=0.05)
    assert derive_bmi(180, 72) == pytest.approx(22.2, abs=0.05)
    # Missing either input → None.
    assert derive_bmi(175, None) is None
    assert derive_bmi(None, 80) is None
    assert derive_bmi(0, 80) is None


def test_build_valid_demographics():
    ctx = build_demographic_context_v1(VALID_DOC, today=TODAY)
    assert ctx.age == 36
    assert ctx.sex == "male"
    assert ctx.height_cm == 175
    assert ctx.weight_kg == 80
    assert ctx.bmi == pytest.approx(26.1, abs=0.05)
    assert ctx.state == "Telangana"
    assert ctx.city == "Hyderabad"


def test_partial_demographics():
    # Only dob + location present (mirrors the real data shape).
    doc = {"dateOfBirth": datetime(2002, 4, 13), "sex": "female", "state": "Kerala", "city": "Kochi"}
    ctx = build_demographic_context_v1(doc, today=TODAY)
    assert ctx.age == 24
    assert ctx.sex == "female"
    assert ctx.height_cm is None
    assert ctx.weight_kg is None
    assert ctx.bmi is None          # can't derive without both
    assert ctx.city == "Kochi"
    assert not ctx.is_empty()


# ---------------------------------------------------------------------------
# Service (fail-open)
# ---------------------------------------------------------------------------

async def test_service_valid_user():
    svc = _service([VALID_DOC])
    ctx = await svc.load("uid-123")
    assert ctx is not None and ctx.age == 36 and ctx.city == "Hyderabad"


async def test_service_missing_user_returns_none():
    svc = _service([VALID_DOC])
    assert await svc.load("does-not-exist") is None


async def test_service_no_user_id_returns_none():
    svc = _service([VALID_DOC])
    assert await svc.load(None) is None
    assert await svc.load("") is None


async def test_service_mongo_unavailable_is_graceful():
    svc = _service([], raise_exc=RuntimeError("mongo down"))
    # Must not raise — degrades to None.
    assert await svc.load("uid-123") is None


async def test_service_disabled_returns_none():
    svc = _service([VALID_DOC], enabled=False)
    assert await svc.load("uid-123") is None


async def test_repository_projection_excludes_sensitive_fields():
    col = _FakeCollection([VALID_DOC])
    repo = DemographicsRepository(col)
    doc = await repo.fetch("uid-123")
    # Only AI-safe fields returned — no email/phone/password/_id.
    assert set(doc).issubset({"dateOfBirth", "sex", "heightCm", "weightKg", "state", "city"})
    assert "email" not in doc and "phone" not in doc and "password" not in doc
    assert col.last_projection.get("_id") == 0


# ---------------------------------------------------------------------------
# Relevance selection
# ---------------------------------------------------------------------------

FULL = DemographicContextV1(age=36, sex="male", height_cm=175, weight_kg=80, bmi=26.1,
                            state="Telangana", city="Hyderabad")


def test_relevance_bmi_query_gets_body_bundle():
    fields = select_relevant_fields(FULL, {"intent": "lifestyle_query"}, "What is a healthy weight for my BMI?")
    assert {"age", "sex", "height_cm", "weight_kg", "bmi"} <= fields


def test_relevance_generic_knowledge_gets_nothing():
    fields = select_relevant_fields(FULL, {"intent": "condition_explanation"}, "What is hypertension?")
    assert fields == set()


def test_relevance_location_query_gets_location():
    fields = select_relevant_fields(FULL, {"intent": "symptom_query"}, "Is dengue common in my area right now?")
    assert {"state", "city"} <= fields


def test_relevance_sex_specific_query():
    fields = select_relevant_fields(FULL, {"intent": "medication_query"}, "Is this drug safe during pregnancy?")
    assert "sex" in fields and "age" in fields


def test_relevance_symptom_query_gets_age_sex_baseline():
    fields = select_relevant_fields(FULL, {"intent": "symptom_query"}, "I have chest pain")
    assert fields == {"age", "sex"}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def test_render_block_only_relevant_fields():
    block = render_demographic_block(FULL, {"intent": "symptom_query"}, "I have chest pain")
    assert "Age: 36" in block and "Sex: male" in block
    # Body/location not selected for a plain symptom query.
    assert "BMI" not in block and "Location" not in block


def test_render_block_empty_for_generic_knowledge():
    assert render_demographic_block(FULL, {"intent": "condition_explanation"}, "What is asthma?") == ""


def test_render_block_none_demographics():
    assert render_demographic_block(None, {"intent": "symptom_query"}, "I have chest pain") == ""


def test_render_block_bmi_query_full_body():
    block = render_demographic_block(FULL, {"intent": "lifestyle_query"}, "Am I overweight for my height?")
    assert "Height: 175 cm" in block and "Weight: 80 kg" in block and "BMI: 26.1" in block


# ---------------------------------------------------------------------------
# Orchestrator integration (prompt injection + resilience without demographics)
# ---------------------------------------------------------------------------

def test_compose_injects_demographic_block_when_present():
    from app.services.orchestration.pipeline import _compose_answer_prompts

    _sys, user = _compose_answer_prompts(
        query="I have chest pain",
        memory_context="", conversation_history="", vector_context="", graph_context="",
        query_type="symptom_query",
        demographic_context="=== PATIENT DEMOGRAPHICS (authoritative, current) ===\nAge: 36\nSex: male",
    )
    assert "PATIENT DEMOGRAPHICS" in user and "Age: 36" in user


def test_compose_omits_demographic_block_when_absent():
    from app.services.orchestration.pipeline import _compose_answer_prompts

    _sys, user = _compose_answer_prompts(
        query="What is asthma?",
        memory_context="", conversation_history="", vector_context="", graph_context="",
        query_type="condition_explanation",
        demographic_context="",
    )
    assert "PATIENT DEMOGRAPHICS" not in user


class _Container:
    def __init__(self, demographics):
        self.demographics = demographics


async def test_orchestrator_demographic_block_disabled_service():
    # /chat + /chat/stream must work when demographics are off — the injection
    # boundary yields "" and the pipeline proceeds unchanged.
    from app.services.orchestration.pipeline import AsyncOrchestrator

    orch = AsyncOrchestrator(_Container(_service([VALID_DOC], enabled=False)))
    block = await orch._demographic_block(user_id="uid-123", analysis={"intent": "symptom_query"}, query="hi")
    assert block == ""


async def test_orchestrator_demographic_block_no_user_id():
    from app.services.orchestration.pipeline import AsyncOrchestrator

    orch = AsyncOrchestrator(_Container(_service([VALID_DOC])))
    assert await orch._demographic_block(user_id=None, analysis={"intent": "symptom_query"}, query="hi") == ""


async def test_orchestrator_demographic_block_mongo_down_is_graceful():
    from app.services.orchestration.pipeline import AsyncOrchestrator

    orch = AsyncOrchestrator(_Container(_service([], raise_exc=RuntimeError("down"))))
    # Must return "" (not raise) so the chat turn continues.
    assert await orch._demographic_block(user_id="uid-123", analysis={"intent": "symptom_query"}, query="hi") == ""


async def test_orchestrator_demographic_block_relevant_injection():
    from app.services.orchestration.pipeline import AsyncOrchestrator

    orch = AsyncOrchestrator(_Container(_service([VALID_DOC])))
    block = await orch._demographic_block(
        user_id="uid-123", analysis={"intent": "lifestyle_query"}, query="Am I overweight for my height?"
    )
    assert "BMI" in block and "Height" in block
