# Episodic Memory Layer — API Reference

Clinically-aware episodic memory for the Enervera medical AI system.
Backed by Pinecone (namespace per `user_id`), isolated from the Postgres
longitudinal memory subsystem.

**Base URL:** `http://localhost:8001` (or wherever you run `uvicorn`)

## Run locally

```powershell
pip install -e ".[api]"
uvicorn episodic.api.app:app --reload --port 8001
```

Required env vars (in `.env`):

```env
GEMINI_API_KEY=...
PINECONE_API_KEY=...
PINECONE_EPISODIC_INDEX_NAME=enervera-episodic   # optional, defaults shown
```

The Pinecone index is auto-created on first startup (serverless, AWS us-east-1, 1024-dim cosine, llama-text-embed-v2).

---

## 1) `POST /episodic/extract`

Extract an `EpisodeCandidate` from a single utterance. Does **not** store.

### Request

```json
{
  "user_id": "patient_8c2f1a",
  "utterance": "I've had chest pain during exercise for the past two weeks, with shortness of breath and fatigue."
}
```

### Response

```json
{
  "user_id": "patient_8c2f1a",
  "summary": "Recurring exertional chest pain with dyspnea and fatigue for two weeks.",
  "category": "symptom",
  "entities": {
    "symptoms": ["chest pain", "shortness of breath", "fatigue"],
    "conditions": [],
    "medications": [],
    "labs": [],
    "body_parts": ["chest"]
  },
  "temporal_data": {
    "duration": "2 weeks",
    "onset": "",
    "frequency": "during exercise",
    "progression": ""
  },
  "severity": "moderate",
  "clinical_priority": "high",
  "confidence": 0.92,
  "source": "user_self_report",
  "embedding_text": "Recurring chest pain during exercise with shortness of breath and fatigue for 2 weeks",
  "metadata": {},
  "store_memory": true
}
```

If the utterance has no clinical content (greeting, small talk), the endpoint returns `null`.

---

## 2) `POST /episodic/store`

Run the full ingest pipeline: extract → contradiction check → clarification triage → store.

### Request (utterance mode)

```json
{
  "user_id": "patient_8c2f1a",
  "utterance": "I've had chest pain during exercise for the past two weeks."
}
```

### Response

```json
{
  "stored": {
    "episode_id": "f0b7e9b1-72d4-4a3b-9c4d-1234567890ab",
    "user_id": "patient_8c2f1a",
    "timestamp": "2026-05-18T15:24:11.124312+00:00",
    "summary": "Recurring exertional chest pain for two weeks.",
    "category": "symptom",
    "entities": { "symptoms": ["chest pain"], "body_parts": ["chest"], "...": "..." },
    "temporal_data": { "duration": "2 weeks", "frequency": "during exercise" },
    "severity": "moderate",
    "clinical_priority": "high",
    "confidence": 0.92,
    "embedding_text": "Recurring chest pain during exercise for 2 weeks",
    "store_memory": true
  },
  "candidate": { "...": "(same fields, without episode_id/timestamp)" },
  "clarification": { "needs_clarification": false, "questions": [] },
  "contradictions": { "user_id": "patient_8c2f1a", "has_contradictions": false, "contradictions": [], "confidence_penalty": 0.0, "triggers_clarification": false }
}
```

### Response — clarification required

When a safety-critical field is missing (e.g. unlocated severe pain), the candidate is **not stored** and `stored` is `null`:

```json
{
  "stored": null,
  "candidate": { "summary": "Severe pain reported with no location specified.", "...": "..." },
  "clarification": {
    "needs_clarification": true,
    "questions": [
      {
        "reason": "missing_location",
        "question": "Where exactly is the pain — chest, abdomen, head, or elsewhere?",
        "safety_critical": true
      }
    ]
  },
  "contradictions": { "...": "..." }
}
```

The caller asks the question, then re-POSTs to `/episodic/store` with the patient's reply concatenated to the original utterance.

### Request (candidate mode — bypass the LLM pipeline)

```json
{
  "user_id": "patient_8c2f1a",
  "candidate": {
    "user_id": "patient_8c2f1a",
    "summary": "Penicillin allergy, anaphylaxis history.",
    "category": "allergy",
    "entities": { "medications": ["penicillin"], "symptoms": ["anaphylaxis"] },
    "severity": "critical",
    "clinical_priority": "critical",
    "confidence": 1.0,
    "embedding_text": "Penicillin allergy with prior anaphylaxis"
  }
}
```

---

## 3) `POST /episodic/retrieve`

Ranked retrieval. Returns the top-N composite-ranked episodes for the query, **without** compression.

### Request

```json
{
  "user_id": "patient_8c2f1a",
  "query_text": "Is my chest pain getting worse?",
  "top_k": 20,
  "return_k": 5,
  "categories": ["symptom", "consultation"],
  "since": "2025-11-01T00:00:00Z"
}
```

`categories`, `since`, `until` are optional metadata filters.

### Response

```json
{
  "user_id": "patient_8c2f1a",
  "query_text": "Is my chest pain getting worse?",
  "episodes": [
    {
      "episode": { "episode_id": "f0b7e9b1-...", "summary": "Exertional chest pain for 2 weeks", "...": "..." },
      "score": 0.812,
      "factors": {
        "similarity": 0.91,
        "recency": 0.95,
        "priority": 0.80,
        "confidence": 0.92,
        "recurrence": 0.50
      }
    }
  ]
}
```

---

## 4) `POST /episodic/clarify`

Standalone clarification triage. Returns at most one question.

### Request

```json
{
  "user_id": "patient_8c2f1a",
  "utterance": "I have been in pain.",
  "candidate": null
}
```

### Response

```json
{
  "needs_clarification": true,
  "questions": [
    {
      "reason": "missing_location",
      "question": "Where is the pain located?",
      "safety_critical": true
    }
  ]
}
```

---

## 5) `POST /episodic/context`

End-to-end: retrieve → rank → compress → return a prompt-ready block. This is the entry point the Retrieval Orchestrator calls.

### Request

```json
{
  "user_id": "patient_8c2f1a",
  "query_text": "Should I be worried about these symptoms?",
  "top_k": 20,
  "return_k": 5
}
```

### Response

```json
{
  "user_id": "patient_8c2f1a",
  "query_text": "Should I be worried about these symptoms?",
  "episodes": [
    { "episode": { "...": "..." }, "score": 0.78, "factors": { "...": "..." } }
  ],
  "compressed": [
    {
      "representative_id": "f0b7e9b1-...",
      "member_ids": ["f0b7e9b1-...", "abc12345-...", "def67890-..."],
      "category": "symptom",
      "summary": "Recurring exertional chest pain over the past month, worsening trend, peak severity moderate, no medication started.",
      "first_seen": "2026-04-18T10:11:00+00:00",
      "last_seen": "2026-05-17T19:42:00+00:00",
      "peak_severity": "moderate",
      "score": 0.81
    }
  ],
  "rendered_prompt": "=== RECURRING / LONG-RUNNING THEMES ===\n- [symptom] Recurring exertional chest pain ... (2026-04-18 → 2026-05-17; peak: moderate)\n\n=== RECENT EPISODIC MEMORY ===\n- 2026-05-15 [high] (moderate) Episode of chest pain after stairs ...",
  "metadata": {
    "strategy": "retrieve+rank+compress",
    "raw_count": 8,
    "kept_count": 3,
    "compressed_count": 1
  }
}
```

The orchestrator drops `rendered_prompt` directly into the answer LLM's context.

---

## 6) `POST /episodic/contradictions`

Detect contradictions between a new claim and the patient's prior episodic memory.

### Request

```json
{
  "user_id": "patient_8c2f1a",
  "new_claim": "I do not have any diabetes — never been diagnosed.",
  "top_k": 10
}
```

### Response

```json
{
  "user_id": "patient_8c2f1a",
  "has_contradictions": true,
  "contradictions": [
    {
      "prior_episode_id": "a1b2c3d4-5678-90ab-cdef-1234567890ab",
      "prior_summary": "Started metformin 500 mg twice daily for type 2 diabetes.",
      "current_claim": "I do not have any diabetes — never been diagnosed.",
      "reason": "Patient previously reported being on metformin for T2DM; current denial of any diabetes contradicts that.",
      "severity": "warning"
    }
  ],
  "confidence_penalty": 0.15,
  "triggers_clarification": true
}
```

When `triggers_clarification=true`, the orchestrator should call `/episodic/clarify` with the contradiction context, then resolve before storing the new claim.

---

## Ranking formula

Composite score (weights from `EpisodicConfig`):

```
score = 0.45 * similarity        # cosine from Pinecone
      + 0.20 * recency_score     # exponential decay, 14-day half-life
      + 0.15 * priority_weight   # critical=1.0, high=0.8, medium=0.55, low=0.3
      + 0.10 * confidence        # episode confidence
      + 0.10 * recurrence_boost  # count-based, within result set
```

Chronic conditions (allergies, undated conditions) **never decay** — their recency component stays at 1.0 forever. Critical-priority episodes decay at half the normal rate (28-day half-life).

## Memory decay

Computed at retrieval time, not stored:

| Episode kind                   | Decay behavior                                   |
|--------------------------------|--------------------------------------------------|
| Allergy                        | Persistent (1.0 forever)                         |
| Chronic condition (no end date)| Persistent (1.0 forever)                         |
| Critical-priority episode      | Half-life 28 days                                |
| Anything else                  | Half-life 14 days, floor at 0.05                 |

Recurring symptoms gain importance through the **recurrence boost** in the ranker — they don't write back to storage.

## Forward compatibility

- Each `Episode` carries a stable `episode_id` UUID. A future Postgres `clinical_fact` row can reference it via a `source_episode_id UUID` column without any FK to this layer.
- `EpisodicRepository` is a `Protocol`. A graph-backed implementation drops in without changing service callers.
- Pinecone metadata schema is forward-compatible: new fields can be added without re-embedding existing vectors.

## Architecture

```
Query Analyzer
     │
     ▼
Episodic Memory Layer        ← THIS MODULE
     │  /episodic/clarify
     │  /episodic/extract
     │  /episodic/store
     │  /episodic/retrieve
     │  /episodic/context
     │  /episodic/contradictions
     ▼
Retrieval Orchestrator
```
