import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_PATH = PROJECT_ROOT / "tests" / "fixtures" / "clinical_benchmark.json"

EXPECTED_SPECIALTIES = {
    "general_medicine",
    "cardiology",
    "pulmonology",
    "orthopaedics",
    "dermatology",
    "ophthalmology",
    "ent",
}

EXPECTED_DIMENSIONS = {
    "clinical_accuracy",
    "red_flag_recognition",
    "risk_classification",
    "escalation_decision",
    "specialty_relevance",
    "follow_up_quality",
    "unsupported_claims",
    "communication_quality",
}

REQUIRED_CASE_FIELDS = {
    "case_id",
    "specialty",
    "user_query",
    "conversation_context",
    "expected_intent",
    "expected_risk_level",
    "expected_final_action",
    "required_response_properties",
    "forbidden_response_properties",
    "critical_failure_conditions",
    "review_status",
    "reviewer_notes",
}


@pytest.fixture
def benchmark():
    with BENCHMARK_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def test_benchmark_has_required_metadata(benchmark):
    assert benchmark["benchmark_name"]
    assert benchmark["version"]
    assert benchmark["status"] == "draft_pending_clinical_review"


def test_benchmark_covers_all_seven_specialties(benchmark):
    assert set(benchmark["specialties"]) == EXPECTED_SPECIALTIES


def test_benchmark_has_expected_evaluation_dimensions(benchmark):
    assert set(benchmark["evaluation_dimensions"]) == EXPECTED_DIMENSIONS


def test_benchmark_cases_have_unique_ids_and_required_fields(benchmark):
    cases = benchmark["cases"]
    case_ids = [case["case_id"] for case in cases]

    assert cases, "Benchmark must contain at least one case."
    assert len(case_ids) == len(set(case_ids)), "Case IDs must be unique."

    for case in cases:
        assert REQUIRED_CASE_FIELDS.issubset(case.keys())
        assert case["specialty"] in EXPECTED_SPECIALTIES
        assert isinstance(case["user_query"], str)
        assert isinstance(case["conversation_context"], list)
        assert case["review_status"] in {"pending", "approved", "rejected"}


def test_pending_cases_do_not_claim_clinical_approval(benchmark):
    for case in benchmark["cases"]:
        if case["review_status"] == "pending":
            assert case["expected_intent"] is None
            assert case["expected_risk_level"] is None
            assert case["expected_final_action"] is None
