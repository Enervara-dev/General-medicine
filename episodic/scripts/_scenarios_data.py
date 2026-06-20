"""
Persona-tailored scenarios for the episodic memory demo harness.

Each scenario is intentionally narrow — it exercises ONE feature of the
layer (retrieve / contradiction / compression / clarification / decay /
ingest) so a human reviewer can match the printed output to the
`expected_behavior` line for that scenario.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Scenario:
    label: str
    feature: str            # tag for grouping in the report
    kind: str               # which service to invoke: retrieve|context|clarify|contradictions|store
    expected_behavior: str  # one-line description for the human reader
    payload: dict[str, Any] # request kwargs for the chosen service


# ---------------------------------------------------------------------------
# cardio_test — recurring chest pain, contradiction risk, allergy persistence
# ---------------------------------------------------------------------------

_CARDIO_SCENARIOS: list[Scenario] = [
    Scenario(
        label="Recall — heart-related history",
        feature="retrieve",
        kind="retrieve",
        expected_behavior="Top results should be chest-pain / ER / ECG / statin episodes. Allergy and chronic HTN may rank lower without recency.",
        payload={"query_text": "Have I had any heart-related issues recently?"},
    ),
    Scenario(
        label="Contradiction — denies cardiac history",
        feature="contradiction",
        kind="contradictions",
        expected_behavior="Should flag contradiction against prior chest-pain + ER episodes. severity ≥ warning, triggers_clarification=true.",
        payload={"new_claim": "I have never had any cardiac symptoms or chest pain in my life.", "top_k": 10},
    ),
    Scenario(
        label="Decay — pull all chest-pain episodes",
        feature="decay",
        kind="retrieve",
        expected_behavior="The 5-day-old and 12-day-old chest-pain episodes should rank above the 75-day-old one (similarity comparable, recency differs).",
        payload={"query_text": "chest pain on exertion", "top_k": 20, "return_k": 8},
    ),
    Scenario(
        label="Allergy persistence — aspirin",
        feature="decay",
        kind="retrieve",
        expected_behavior="Aspirin allergy episode (~800 days old) should surface with factors.recency == 1.0 (chronic, no decay).",
        payload={"query_text": "what am I allergic to?"},
    ),
    Scenario(
        label="Compression — chest pain history",
        feature="compression",
        kind="context",
        expected_behavior="≥3 chest-pain episodes should collapse into 1 CompressedEpisode with window dates and peak severity.",
        payload={"query_text": "tell me about my chest pain history", "top_k": 20, "return_k": 8},
    ),
]


# ---------------------------------------------------------------------------
# migraine_test — compression target, propranolol contradiction
# ---------------------------------------------------------------------------

_MIGRAINE_SCENARIOS: list[Scenario] = [
    Scenario(
        label="Compression — recurring migraines",
        feature="compression",
        kind="context",
        expected_behavior="The 5 migraine episodes should collapse into 1 CompressedEpisode. compressed_count >= 1, member_ids length >= 3.",
        payload={"query_text": "my migraine pattern over the last two months", "top_k": 20, "return_k": 8},
    ),
    Scenario(
        label="Clarification — ambiguous headache",
        feature="clarify",
        kind="clarify",
        expected_behavior="Vague utterance with no severity/location should yield exactly 1 clarification question (or 0 if the model decides existing context is enough).",
        payload={"utterance": "I had a bad headache yesterday."},
    ),
    Scenario(
        label="Contradiction — claims still on propranolol",
        feature="contradiction",
        kind="contradictions",
        expected_behavior="Should flag contradiction vs the 'propranolol discontinued' episode. severity warning, triggers_clarification=true.",
        payload={"new_claim": "I'm currently taking propranolol daily for my migraines.", "top_k": 10},
    ),
    Scenario(
        label="Recall — migraine medications tried",
        feature="retrieve",
        kind="retrieve",
        expected_behavior="Propranolol-start and propranolol-stop episodes should rank above unrelated lab/lifestyle entries.",
        payload={"query_text": "what medications have I tried for migraines?"},
    ),
]


# ---------------------------------------------------------------------------
# geriatric_test — chronic persistence, polypharmacy, decay
# ---------------------------------------------------------------------------

_GERIATRIC_SCENARIOS: list[Scenario] = [
    Scenario(
        label="Chronic persistence — long-term conditions",
        feature="decay",
        kind="retrieve",
        expected_behavior="HTN / T2DM / CKD condition episodes should surface with factors.recency == 1.0 despite being years old.",
        payload={"query_text": "what are my long-term medical conditions?"},
    ),
    Scenario(
        label="Polypharmacy — current medications",
        feature="retrieve",
        kind="retrieve",
        expected_behavior="At least 4-5 distinct medication episodes (lisinopril, metformin, atorvastatin, aspirin, vitamin D) should be returned.",
        payload={"query_text": "what medications am I currently taking?", "return_k": 8},
    ),
    Scenario(
        label="Allergy persistence — penicillin",
        feature="decay",
        kind="retrieve",
        expected_behavior="Penicillin allergy episode (~3000 days old) should surface with factors.recency == 1.0.",
        payload={"query_text": "any drug allergies I have?"},
    ),
    Scenario(
        label="Decay — fall history vs recent dizziness",
        feature="decay",
        kind="context",
        expected_behavior="90-day-old fall should still be visible but ranked below the recent dizziness episode for a balance-related query.",
        payload={"query_text": "history of falls or balance issues", "top_k": 20, "return_k": 8},
    ),
    Scenario(
        label="End-to-end ingest — new lab value",
        feature="store",
        kind="store",
        expected_behavior="Should extract a lab episode (glucose 312), store it, no clarifications needed, no contradictions (consistent with T2DM history).",
        payload={"utterance": "My blood sugar was 312 this morning before breakfast."},
    ),
]


SCENARIOS_BY_PERSONA: dict[str, list[Scenario]] = {
    "cardio_test": _CARDIO_SCENARIOS,
    "migraine_test": _MIGRAINE_SCENARIOS,
    "geriatric_test": _GERIATRIC_SCENARIOS,
}
