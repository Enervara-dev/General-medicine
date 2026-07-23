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


def test_relevance_direct_profile_asks():
    # "which city am I from?" must inject location even without the phrase "my city".
    assert {"city", "state"} <= select_relevant_fields(
        FULL, {"intent": "followup_query"}, "which city am I from?")
    # "how old am I?" injects age.
    assert "age" in select_relevant_fields(FULL, {"intent": "followup_query"}, "how old am I?")
    # "what's my BMI?" pulls the body bundle.
    assert {"bmi", "height_cm", "weight_kg"} <= select_relevant_fields(
        FULL, {"intent": "followup_query"}, "what's my BMI?")


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
# Fix 2: Mongo demographics authoritative over conversational state
# ---------------------------------------------------------------------------

def _state_with_conv_demo(age=None, sex=None, name=None):
    from Memory_Layer.session_memory.models import StructuredState
    demo = {}
    if age is not None:
        demo["age"] = age
    if sex is not None:
        demo["sex"] = sex
    if name is not None:
        demo["name"] = name
    return StructuredState(demographics=demo, symptoms=["fever"])


def test_conversational_age_sex_suppressed_when_mongo_authoritative():
    from Memory_Layer.session_memory.context_builder import _state_lines
    # Conversation claims 40/female; Mongo owns age+sex → suppress the claims.
    state = _state_with_conv_demo(age=40, sex="female")
    lines = "\n".join(_state_lines(state, authoritative_demographics={"age", "sex"}))
    assert "40" not in lines
    assert "female" not in lines
    # The clinical fact is still rendered (we only suppress demographics).
    assert "fever" in lines.lower()


def test_conversational_demo_labeled_unverified_when_no_mongo():
    from Memory_Layer.session_memory.context_builder import _state_lines
    # No authoritative fields (anonymous / incomplete profile) → keep but label.
    state = _state_with_conv_demo(age=40, sex="female")
    lines = "\n".join(_state_lines(state, authoritative_demographics=frozenset()))
    assert "unverified" in lines.lower()
    assert "40" in lines and "female" in lines


def test_name_kept_even_when_demographics_authoritative():
    from Memory_Layer.session_memory.context_builder import _state_lines
    # Mongo has no name; the conversational name is always preserved.
    state = _state_with_conv_demo(age=40, sex="female", name="Aarav")
    lines = "\n".join(_state_lines(state, authoritative_demographics={"age", "sex"}))
    assert "Aarav" in lines
    assert "40" not in lines  # age still suppressed


def test_partial_authoritative_only_suppresses_owned_field():
    from Memory_Layer.session_memory.context_builder import _state_lines
    # Mongo owns age but not sex → suppress age, keep sex (labeled unverified).
    state = _state_with_conv_demo(age=40, sex="female")
    lines = "\n".join(_state_lines(state, authoritative_demographics={"age"}))
    assert "40" not in lines
    assert "female" in lines and "unverified" in lines.lower()


def test_build_memory_context_threads_authoritative():
    from Memory_Layer.session_memory.context_builder import build_memory_context
    from Memory_Layer.session_memory.retriever import get_working_memory
    from Memory_Layer.session_memory.models import SessionMemory
    s = SessionMemory(session_id="s")
    s.state = _state_with_conv_demo(age=40, sex="male")
    wm = get_working_memory(s)
    ctx = build_memory_context(wm, authoritative_demographics={"age", "sex"})
    assert "40" not in ctx  # conversational age suppressed end-to-end


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


async def test_orchestrator_demographics_disabled_service():
    # /chat + /chat/stream must work when demographics are off — the loader
    # yields None and the pipeline proceeds unchanged.
    from app.services.orchestration.pipeline import (
        AsyncOrchestrator,
        _authoritative_demographic_fields,
    )

    orch = AsyncOrchestrator(_Container(_service([VALID_DOC], enabled=False)))
    demo = await orch._load_demographics("uid-123")
    assert demo is None
    assert _authoritative_demographic_fields(demo) == frozenset()
    assert render_demographic_block(demo, {"intent": "symptom_query"}, "hi") == ""


async def test_orchestrator_demographics_no_user_id():
    from app.services.orchestration.pipeline import AsyncOrchestrator

    orch = AsyncOrchestrator(_Container(_service([VALID_DOC])))
    assert await orch._load_demographics(None) is None


async def test_orchestrator_demographics_mongo_down_is_graceful():
    from app.services.orchestration.pipeline import AsyncOrchestrator

    orch = AsyncOrchestrator(_Container(_service([], raise_exc=RuntimeError("down"))))
    # Must return None (not raise) so the chat turn continues.
    assert await orch._load_demographics("uid-123") is None


async def test_orchestrator_demographics_relevant_injection_and_authority():
    from app.services.orchestration.pipeline import (
        AsyncOrchestrator,
        _authoritative_demographic_fields,
    )

    orch = AsyncOrchestrator(_Container(_service([VALID_DOC])))
    demo = await orch._load_demographics("uid-123")
    assert demo is not None
    # Mongo owns age + sex → they become authoritative (suppress conversational).
    assert _authoritative_demographic_fields(demo) == frozenset({"age", "sex"})
    block = render_demographic_block(demo, {"intent": "lifestyle_query"}, "Am I overweight for my height?")
    assert "BMI" in block and "Height" in block
