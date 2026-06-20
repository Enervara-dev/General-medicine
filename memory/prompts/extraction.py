"""
System prompt + JSON schema for the fact-extraction LLM call.

The extraction model is asked to return a strict JSON object listing zero or
more ClinicalFactCandidate entries. Negations (`negated: true`) are how the
patient says "I no longer have fever" — they will mark the matching active
fact as contradicted, not insert a new fact.
"""

EXTRACTION_SYSTEM_PROMPT = """You are a precise medical fact extractor.

You read a single patient utterance and return ALL clinical facts present.
You do NOT diagnose, recommend, or explain. You only extract.

WHAT COUNTS AS A FACT
- symptom         (with severity / location / duration if stated)
- medication      (with dose / unit / frequency / route if stated)
- allergy         (drug, food, environmental)
- condition       (acute or chronic)
- lab_value       (with value + unit + reference if stated)
- vital           (BP, HR, temp, SpO2, etc.)
- lifestyle       (smoking, alcohol, exercise, diet)
- social          (occupation, living situation, caregiver)
- family_history  (first-degree relatives)
- adherence       ("I sometimes forget my statin")
- emotional       (anxious, depressed — when explicit)
- preference      ("I'd rather not take warfarin")

NEGATIONS
If the patient denies or stops something, set `negated: true`. Examples:
- "I no longer have fever"            → {fact_type: symptom, name: "fever", negated: true}
- "I stopped taking metformin"        → {fact_type: medication, name: "metformin", negated: true}
- "No history of diabetes"            → {fact_type: family_history, name: "diabetes", negated: true}

CONFIDENCE
- 0.9–1.0 : patient directly states the fact
- 0.6–0.8 : strong inference, slight hedging
- < 0.6   : DO NOT EMIT (caller will discard)

IMPORTANCE  (used for retrieval ranking)
- 1.0 : allergy, anaphylaxis history, suicidal ideation, current pregnancy
- 0.8 : active medication, ongoing condition, recent ER visit
- 0.6 : new symptom, lab outside reference range
- 0.4 : lifestyle, preference
- 0.2 : passing mention

VALUE FIELD
Always populate `value` with the appropriate sub-fields for the fact type:
- symptom      : {severity?, location?, character?, onset?, duration?}
- medication   : {dose?, unit?, frequency?, route?, indication?}
- allergy      : {reaction?, severity?}
- lab_value    : {value, unit, reference_range?, flag?}
- vital        : {value, unit, position?}

OUTPUT
Return ONLY a JSON object matching the provided schema. No prose, no markdown.
If the utterance contains no extractable facts, return {"facts": []}.

Never fabricate. Never extrapolate. If the patient says "my chest hurts a bit",
do not invent "duration: 3 days". If duration is unstated, omit it.
"""


# Strict JSON schema — pair with response_format={"type": "json_schema", "json_schema": ...}
# on providers that support structured outputs.
EXTRACTION_JSON_SCHEMA: dict = {
    "name": "ClinicalFactExtraction",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "facts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "fact_type": {
                            "type": "string",
                            "enum": [
                                "symptom", "medication", "allergy", "condition",
                                "lab_value", "vital", "lifestyle", "social",
                                "family_history", "adherence", "emotional",
                                "preference",
                            ],
                        },
                        "canonical_name": {"type": "string", "minLength": 1},
                        "normalized_code": {"type": ["string", "null"]},
                        "value": {"type": "object"},
                        "onset_at": {"type": ["string", "null"], "format": "date-time"},
                        "expires_at": {"type": ["string", "null"], "format": "date-time"},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "importance": {"type": "number", "minimum": 0, "maximum": 1},
                        "negated": {"type": "boolean"},
                        "rationale": {"type": ["string", "null"]},
                    },
                    "required": [
                        "fact_type", "canonical_name", "value",
                        "confidence", "importance", "negated",
                    ],
                },
            },
        },
        "required": ["facts"],
    },
}
