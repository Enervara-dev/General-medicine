# Longitudinal Patient Memory — Architecture Redesign

## 1. Why This Rebuild

The current memory layer (`Memory_Layer/session_memory/`) treats raw conversation turns + a rolling prose summary as the unit of memory. That works for short sessions but fails for a real medical assistant:

- A turn from three weeks ago contradicting today's turn is invisible — both end up in the summary, blended together.
- Allergies, active medications, and unresolved follow-ups have no first-class representation; they live inside list fields on a Pydantic model with no provenance, no expiry, no contradiction handling.
- Redis is the canonical store. When TTL expires (2 hours by default), the *patient* — not the session — loses their memory.
- Every turn re-injects the same prose summary into the prompt; we are paying tokens to re-tell the LLM what it already inferred.
- There is no audit trail. We cannot answer "why did the assistant say X?" after the fact.

**This redesign treats *extracted structured clinical facts* as the unit of memory.** Conversations become an *audit log*, never a prompt input. Facts have temporal validity, confidence, importance, provenance, and supersede chains. Postgres is canonical; Redis is a hot cache; pgvector is a *narrow* retrieval channel for nuanced prose only when structured retrieval cannot satisfy a query.

---

## 2. Design Principles

| # | Principle | Concrete Implication |
|---|---|---|
| 1 | **Facts, not turns, are memory** | `clinical_fact` table is the unit; `conversation_event` is audit-only and never enters a prompt. |
| 2 | **Temporal validity is first-class** | Every fact has `onset_at`, `observed_at`, `expires_at`, `status`. Stale meds drop off automatically. |
| 3 | **Postgres is canonical** | All durable state lives in Postgres. Redis can be wiped without data loss. |
| 4 | **Structured before semantic** | Retrieval always queries the SQL tables first. pgvector is a *fallback* for nuance, capped at top-2. |
| 5 | **Provenance is mandatory** | Every fact carries `source_event_id`. Every prompt is logged with the fact IDs it consumed (`retrieval_log`). |
| 6 | **Contradictions are detected, not blended** | New facts that conflict mark the old as `superseded` or `contradicted` and chain via `supersedes_id`. |
| 7 | **Importance and decay drive prioritization** | Allergies never decay; chitchat decays fast; recent ER visit > old check-up. |
| 8 | **Compress, don't replay** | "Current Patient State" is a continuously regenerated paragraph + structured snapshot. Raw turns never re-enter the prompt. |
| 9 | **Async-native** | Every service exposes `async def`; SQLAlchemy 2.x async engine; Redis via `redis.asyncio`. FastAPI-ready. |
| 10 | **Safety gates are explicit code paths** | Allergy collision and drug-interaction checks run *before* the LLM call. |

**Anti-patterns we explicitly avoid:**

- "Embed every message" architectures — they bloat the index, retrieve noise, and have no temporal model.
- "Replay the transcript" — recent turns past the immediate one go through extraction, not into the prompt.
- "Summary as memory" — summaries are *generated from* facts, not the other way around.
- Vector-only memory — pgvector solves a narrow problem (recall on nuance); SQL solves the rest.

---

## 3. Storage Tier Responsibilities

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Redis (working memory / hot cache)                                      │
│  - patient_state snapshot JSON       TTL 5 min, invalidated on write     │
│  - "Current Patient State" prose     TTL 5 min                           │
│  - request-scoped retrieval cache    TTL 60 s                            │
│  - rate-limit counters, idempotency keys                                 │
│  REDIS CAN BE FLUSHED AT ANY TIME WITHOUT DATA LOSS                      │
├──────────────────────────────────────────────────────────────────────────┤
│  PostgreSQL (canonical longitudinal store)                               │
│  - patient                  (identity)                                   │
│  - clinical_fact            (atomic facts; the unit of memory)           │
│  - patient_state            (denormalized current snapshot)              │
│  - episodic_memory          (clinically-significant events)              │
│  - conversation_event       (audit-only; never prompted)                 │
│  - retrieval_log            (provenance of every LLM call)               │
│  ALL JSONB FIELDS GIN-INDEXED                                            │
├──────────────────────────────────────────────────────────────────────────┤
│  PostgreSQL + pgvector (semantic recall, narrow channel)                 │
│  - semantic_memory          (nuanced prose snippets; HNSW vector index)  │
│  USED ONLY WHEN STRUCTURED RETRIEVAL CANNOT ANSWER A QUERY               │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Database Schema

### 4.1 `patient`

```sql
CREATE TABLE patient (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    external_id     VARCHAR(128) UNIQUE,        -- e.g. EHR identifier
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata        JSONB NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX patient_external_id_idx ON patient (external_id);
```

### 4.2 `conversation_event` (audit only — never prompted)

```sql
CREATE TYPE conv_role AS ENUM ('user', 'assistant', 'system');

CREATE TABLE conversation_event (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id          UUID NOT NULL REFERENCES patient(id) ON DELETE CASCADE,
    session_id          VARCHAR(128) NOT NULL,
    role                conv_role NOT NULL,
    content             TEXT NOT NULL,
    analysis_payload    JSONB,                  -- gatekeeper output for audit
    request_id          UUID,                   -- correlates user+assistant pair
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX conversation_event_patient_time_idx
    ON conversation_event (patient_id, created_at DESC);
CREATE INDEX conversation_event_session_idx
    ON conversation_event (session_id, created_at);
```

### 4.3 `clinical_fact` (the unit of memory)

```sql
CREATE TYPE fact_type AS ENUM (
    'symptom', 'medication', 'allergy', 'condition',
    'lab_value', 'vital', 'lifestyle', 'social',
    'family_history', 'adherence', 'emotional', 'preference'
);
CREATE TYPE fact_status AS ENUM (
    'active', 'resolved', 'superseded', 'contradicted', 'refuted'
);
CREATE TYPE fact_source AS ENUM (
    'patient_report', 'llm_extraction', 'lab_import', 'manual_entry'
);

CREATE TABLE clinical_fact (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id          UUID NOT NULL REFERENCES patient(id) ON DELETE CASCADE,
    fact_type           fact_type NOT NULL,
    canonical_name      VARCHAR(256) NOT NULL,           -- e.g. "chest pain"
    normalized_code     VARCHAR(64),                     -- SNOMED / RxNorm / ICD-10
    value               JSONB NOT NULL DEFAULT '{}'::jsonb,
                                                          -- typed per fact_type
                                                          -- medication: {dose, unit, frequency, route}
                                                          -- symptom:    {severity, location, character}
                                                          -- lab_value:  {value, unit, reference_range}

    onset_at            TIMESTAMPTZ,                     -- when the fact became true
    observed_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at          TIMESTAMPTZ,                     -- NULL = ongoing
    status              fact_status NOT NULL DEFAULT 'active',

    confidence          REAL NOT NULL DEFAULT 0.8,       -- 0.0–1.0
    importance          REAL NOT NULL DEFAULT 0.5,       -- 0.0–1.0; allergies = 1.0
    decay_score         REAL NOT NULL DEFAULT 1.0,       -- recomputed by DecayService

    source              fact_source NOT NULL,
    source_event_id     UUID REFERENCES conversation_event(id) ON DELETE SET NULL,
    supersedes_id       UUID REFERENCES clinical_fact(id) ON DELETE SET NULL,
    contradicts_id      UUID REFERENCES clinical_fact(id) ON DELETE SET NULL,

    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Hot path: "give me all active facts of type X for patient Y"
CREATE INDEX clinical_fact_active_idx
    ON clinical_fact (patient_id, fact_type, status, importance DESC)
    WHERE status = 'active';

-- For expiry sweeps
CREATE INDEX clinical_fact_expiry_idx
    ON clinical_fact (expires_at)
    WHERE status = 'active' AND expires_at IS NOT NULL;

-- For JSONB attribute queries (e.g. value->>'severity' = 'severe')
CREATE INDEX clinical_fact_value_gin ON clinical_fact USING GIN (value);

-- For provenance lookups
CREATE INDEX clinical_fact_source_event_idx ON clinical_fact (source_event_id);
```

### 4.4 `patient_state` (denormalized snapshot — fast read)

```sql
CREATE TABLE patient_state (
    patient_id          UUID PRIMARY KEY REFERENCES patient(id) ON DELETE CASCADE,
    snapshot            JSONB NOT NULL,                  -- structured by fact_type
                                                          -- {
                                                          --   "symptoms":    [...],
                                                          --   "medications": [...],
                                                          --   "allergies":   [...],
                                                          --   "conditions":  [...]
                                                          -- }
    summary_text        TEXT,                            -- "Current Patient State" prose
    risk_level          VARCHAR(16) NOT NULL DEFAULT 'none',
    last_consolidated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    version             INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX patient_state_snapshot_gin ON patient_state USING GIN (snapshot);
```

### 4.5 `episodic_memory`

```sql
CREATE TYPE episode_type AS ENUM (
    'er_visit', 'hospitalization', 'adverse_drug_reaction',
    'new_diagnosis', 'symptom_onset', 'treatment_change', 'milestone'
);

CREATE TABLE episodic_memory (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id          UUID NOT NULL REFERENCES patient(id) ON DELETE CASCADE,
    title               VARCHAR(256) NOT NULL,
    description         TEXT NOT NULL,
    event_type          episode_type NOT NULL,
    occurred_at         TIMESTAMPTZ NOT NULL,
    recorded_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    importance          REAL NOT NULL DEFAULT 0.5,
    related_fact_ids    UUID[] NOT NULL DEFAULT '{}',
    source_event_id     UUID REFERENCES conversation_event(id) ON DELETE SET NULL
);
CREATE INDEX episodic_memory_time_idx
    ON episodic_memory (patient_id, occurred_at DESC);
```

### 4.6 `semantic_memory` (pgvector — narrow channel)

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE semantic_memory (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id          UUID NOT NULL REFERENCES patient(id) ON DELETE CASCADE,
    content             TEXT NOT NULL,                   -- the nuanced passage
    embedding           VECTOR(1024) NOT NULL,           -- llama-text-embed-v2 dim
    importance          REAL NOT NULL DEFAULT 0.5,
    decay_score         REAL NOT NULL DEFAULT 1.0,
    last_accessed_at    TIMESTAMPTZ,
    access_count        INTEGER NOT NULL DEFAULT 0,
    source_event_id     UUID REFERENCES conversation_event(id) ON DELETE SET NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX semantic_memory_patient_idx ON semantic_memory (patient_id);
CREATE INDEX semantic_memory_embedding_hnsw
    ON semantic_memory USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
```

### 4.7 `retrieval_log` (provenance per LLM call)

```sql
CREATE TABLE retrieval_log (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id              UUID NOT NULL REFERENCES patient(id) ON DELETE CASCADE,
    request_id              UUID NOT NULL,
    query_text              TEXT NOT NULL,
    routing_mode            VARCHAR(32) NOT NULL,       -- no_retrieval / memory_first / hybrid_rag
    retrieved_fact_ids      UUID[] NOT NULL DEFAULT '{}',
    retrieved_episode_ids   UUID[] NOT NULL DEFAULT '{}',
    retrieved_semantic_ids  UUID[] NOT NULL DEFAULT '{}',
    retrieval_strategy      VARCHAR(128),               -- human-readable
    prompt_tokens_in        INTEGER,
    completion_tokens_out   INTEGER,
    answer_text             TEXT,                       -- optional snapshot
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX retrieval_log_patient_time_idx
    ON retrieval_log (patient_id, created_at DESC);
CREATE INDEX retrieval_log_request_idx ON retrieval_log (request_id);
```

---

## 5. New Folder Structure

```
memory/
├── __init__.py
├── adapter.py                 # LongitudinalMemoryAdapter — single entry point
├── db/
│   ├── __init__.py
│   ├── base.py                # DeclarativeBase, naming conventions
│   ├── session.py             # async engine, AsyncSession factory
│   └── migrations/            # alembic
│       ├── env.py
│       └── versions/
│           └── 0001_initial.py
├── models/                    # SQLAlchemy ORM
│   ├── __init__.py
│   ├── patient.py
│   ├── conversation_event.py
│   ├── clinical_fact.py
│   ├── patient_state.py
│   ├── episodic_memory.py
│   ├── semantic_memory.py
│   └── retrieval_log.py
├── schemas/                   # Pydantic DTOs
│   ├── __init__.py
│   ├── fact.py
│   ├── state.py
│   └── retrieval.py
├── services/                  # Application services (async)
│   ├── __init__.py
│   ├── extraction.py          # turn -> ClinicalFact candidates
│   ├── consolidation.py       # merge / supersede / contradict
│   ├── retrieval.py           # build MemoryContext
│   ├── summarization.py       # snapshot -> prose
│   ├── safety.py              # allergy collision, contradiction surfacing
│   └── decay.py               # background importance recompute
├── cache/
│   ├── __init__.py
│   └── redis_cache.py         # hot snapshot cache
├── prompts/
│   ├── __init__.py
│   ├── extraction.py          # extraction LLM system prompt
│   └── summarization.py
├── pipelines/
│   └── update_after_turn.py   # orchestrates extraction → consolidation → cache invalidation
└── api/
    └── deps.py                # FastAPI dependency injection helpers
```

`Memory_Layer/session_memory/` stays in place during migration and is deleted at the end of Phase E (see §11).

---

## 6. Memory Lifecycle

```
                          ┌──────────────────┐
   1. User turn ─────────▶│ ConversationEvent│ (persisted, never prompted)
                          └────────┬─────────┘
                                   │
                                   ▼
                          ┌──────────────────┐
                          │ ExtractionService│  LLM (small, fast, JSON-strict)
                          │  prompt + turn → │  → list[ClinicalFactCandidate]
                          │  fact candidates │
                          └────────┬─────────┘
                                   │
                                   ▼
                          ┌──────────────────┐
                          │   SafetyService  │  Allergy / interaction / contradiction
                          │  pre-flight risk │  → emit RiskFlag events
                          └────────┬─────────┘
                                   │
                                   ▼
                          ┌──────────────────┐
                          │ConsolidationSvc  │  Merge candidates into clinical_fact:
                          │  - dedupe        │   - existing match → bump observed_at
                          │  - supersede     │   - same name diff value → supersede
                          │  - contradict    │   - "no fever" vs active "fever" → contradict
                          │  - new insert    │
                          └────────┬─────────┘
                                   │
                                   ▼
                          ┌──────────────────┐
                          │ State Snapshot   │  Recompute patient_state.snapshot JSONB
                          │  + Summary       │  Regenerate summary_text if Δ above threshold
                          └────────┬─────────┘
                                   │
                                   ▼
                          ┌──────────────────┐
                          │ Redis cache      │  Invalidate snapshot key
                          │  invalidate      │
                          └──────────────────┘

   (Background, hourly)   ┌──────────────────┐
                          │ DecayService     │  recompute decay_score for facts +
                          │  expiry sweep    │  semantic_memory; set status='resolved'
                          │                  │  where expires_at < now()
                          └──────────────────┘
```

---

## 7. Retrieval Flow

```
   User turn arriving at Stage 0 of pipeline
                          │
                          ▼
            ┌─────────────────────────────┐
            │ RetrievalService.build()    │
            └────────────┬────────────────┘
                         │
                         ▼
            ┌─────────────────────────────┐
            │ Hot cache hit?              │
            │ Redis: patient_state:{id}   │
            └─────┬───────────────┬───────┘
                  │ HIT           │ MISS
                  ▼               ▼
            ┌─────────┐   ┌────────────────────────────┐
            │ Return  │   │ Structured query (Postgres)│
            │ cached  │   │ 1. ALL active allergies    │   (always — safety)
            └─────────┘   │ 2. Active medications      │
                          │ 3. Active symptoms ordered │
                          │    by importance, decay    │
                          │ 4. Unresolved follow-ups   │
                          │ 5. Recent episodes (≤30 d) │
                          └────────────┬───────────────┘
                                       │
                                       ▼
                          ┌────────────────────────────┐
                          │ Did structured cover the   │
                          │ query? (intent + entity    │
                          │ overlap heuristic)         │
                          └────────────┬───────────────┘
                                       │
                          ┌────────────┴───────────────┐
                          │ NO                         │ YES
                          ▼                            ▼
            ┌─────────────────────────────┐   ┌──────────────────┐
            │ Vector recall (pgvector)    │   │ Skip vector      │
            │ - embed query               │   │ recall           │
            │ - similarity > 0.75         │   └──────────────────┘
            │ - top_k = 2 max             │
            │ - filter patient_id +       │
            │   decay_score > 0.3         │
            └────────────┬────────────────┘
                         │
                         ▼
            ┌─────────────────────────────┐
            │ Compose MemoryContext       │
            │ - token-budgeted            │
            │ - allergies never trimmed   │
            │ - meds never trimmed        │
            │ - symptoms trimmed by       │
            │   importance × decay_score  │
            └────────────┬────────────────┘
                         │
                         ▼
            ┌─────────────────────────────┐
            │ Write retrieval_log         │  request_id, fact_ids,
            │                             │  strategy, token count
            └────────────┬────────────────┘
                         │
                         ▼
            ┌─────────────────────────────┐
            │ Return MemoryContext to     │
            │ GraphRAGPipeline            │
            └─────────────────────────────┘
```

**Critical rule:** vector recall is bounded by `top_k=2`. We never inject more than two nuance snippets, and only if structured retrieval did not match the query's entities.

---

## 8. Prompt Assembly Redesign

The prompt the LLM sees is *not* "last 6 turns + summary". It is:

```
=== CURRENT PATIENT STATE ===
{patient_state.summary_text}              ← 2-paragraph prose, regenerated on consolidation

=== ACTIVE CLINICAL FACTS ===
Allergies:        {comma-separated list with severity, NEVER OMITTED}
Medications:      {name, dose, frequency — one per line}
Active symptoms:  {name, onset, severity — sorted by importance × decay_score}
Unresolved:       {short list of follow-ups the patient hasn't answered}
Conditions:       {chronic conditions list}

=== RECENT EPISODES ===
{up to 3 episodic_memory entries, last 30 days, by importance × recency}

=== RELEVANT PRIOR DISCUSSION ===
{0–2 semantic_memory snippets, only present if query needed nuance}

=== USER QUESTION ===
{the current turn}
```

What is **never** in the prompt:
- Any prior turn from `conversation_event`.
- Anything older than 30 days unless surfaced via importance + structured rules.
- Speculative facts (`confidence < 0.6`).
- Superseded or contradicted facts.

This caps prompt growth at O(active facts) regardless of session length. A patient with a 6-month history pays roughly the same tokens as a brand-new patient.

---

## 9. Code Skeletons

See the following new files (production-grade skeletons; fill in `# TODO` blocks):

| Path | Purpose |
|---|---|
| `memory/db/base.py`                          | `DeclarativeBase` with naming conventions |
| `memory/db/session.py`                       | Async engine + `get_session()` dependency |
| `memory/models/*.py`                         | SQLAlchemy ORM for every table above |
| `memory/schemas/{fact,state,retrieval}.py`   | Pydantic DTOs for API + service boundaries |
| `memory/services/extraction.py`              | LLM-driven fact extraction with JSON schema |
| `memory/services/consolidation.py`           | Merge / supersede / contradict logic |
| `memory/services/retrieval.py`               | Structured-first, vector-last assembly |
| `memory/services/safety.py`                  | Allergy + interaction + contradiction gates |
| `memory/services/summarization.py`           | "Current Patient State" prose regen |
| `memory/cache/redis_cache.py`                | Hot snapshot cache with invalidation |
| `memory/adapter.py`                          | `LongitudinalMemoryAdapter` — drop-in for `SessionMemoryAdapter` |
| `memory/prompts/extraction.py`               | The extraction LLM system prompt |

---

## 10. Refactoring Map

| Old (`Memory_Layer/session_memory/`) | New (`memory/`) | Disposition |
|---|---|---|
| `session_manager.py` | `db/session.py` + Postgres | **Replaced**; Redis becomes cache only |
| `models.py` `SessionMemory`, `Message`, `Role`, `RiskLevel` | `models/conversation_event.py`, `models/patient.py` | `Message` becomes `ConversationEvent`; `SessionMemory` deleted |
| `models.py` `StructuredState` | `models/patient_state.py` + `models/clinical_fact.py` | Split — snapshot becomes `patient_state.snapshot`, atoms become `clinical_fact` rows |
| `state_extractor.py` (regex extraction) | `services/extraction.py` (LLM extraction) | **Replaced**; regex kept as cheap fallback |
| `summarizer.py` (turn → summary) | `services/summarization.py` (state → prose) | **Inverted** — summary is now derived from structured state, not from turns |
| `retriever.py` `WorkingMemory` | `schemas/state.py` `MemoryContext` | `WorkingMemory` becomes a Pydantic DTO returned by the retrieval service |
| `context_builder.py` `assemble_context_payload` | `services/retrieval.py` + new prompt assembly in `adapter.py` | Token budgeting moves into retrieval service so facts are scored before assembly |
| `graphrag/memory/session_adapter.py` | `memory/adapter.py` `LongitudinalMemoryAdapter` | Drop-in replacement; same public surface |

---

## 11. Migration Strategy

A safe four-phase migration that keeps the system serving traffic the whole time.

### Phase A — Add Postgres + dual-write

1. Stand up Postgres + pgvector (Docker compose for local; managed RDS / Cloud SQL for prod).
2. Run Alembic `0001_initial` migration.
3. Wire `LongitudinalMemoryAdapter` alongside the existing `SessionMemoryAdapter`.
4. Every turn writes a `ConversationEvent` to Postgres in addition to Redis. Nothing reads from Postgres yet.
5. Verify writes for a week, monitor lag.

### Phase B — Backfill from Redis

1. For each active Redis session, run a one-off backfill:
   - Read `SessionMemory.recent_turns` → upsert `ConversationEvent` rows.
   - Read `SessionMemory.state` → split into `ClinicalFact` rows (one per symptom/drug/allergy/condition).
   - Mark `source = 'patient_report'`, `confidence = 0.7`, `importance = 0.6`, `source_event_id = NULL` (legacy provenance).
2. Idempotent — backfill can re-run if interrupted.

### Phase C — Turn on extraction + consolidation

1. After every turn, enqueue `update_after_turn(patient_id, conversation_event_id)` in a background task (asyncio task in-process for now; Celery/Arq later).
2. `ExtractionService` produces fact candidates; `ConsolidationService` merges them.
3. Reads still come from Redis-backed `SessionMemoryAdapter`. Postgres is being built in parallel.

### Phase D — Switch reads to Postgres

1. Behind a feature flag `LONGITUDINAL_MEMORY_ENABLED`, swap `SessionMemoryAdapter` for `LongitudinalMemoryAdapter` in `GraphRAGPipeline.__init__`.
2. Enable for a small percentage of sessions first. Verify retrieval log shows reasonable fact selection.
3. Compare answer quality side-by-side using `tests/StressTest/run_stress_test.py`.
4. Ramp to 100%.

### Phase E — Decommission Redis as source of truth

1. Redis demoted to cache only (hot snapshot, 5-min TTL).
2. `Memory_Layer/session_memory/{session_manager,summarizer,state_extractor,retriever,context_builder,models}.py` deleted.
3. `graphrag/memory/session_adapter.py` deleted.
4. `RuntimeConfig.SESSION_TTL_SEC` deprecated (now a cache TTL, much smaller).

Rollback at any phase by disabling the feature flag — Redis stays authoritative until Phase E.

---

## 12. Background Jobs

| Job | Schedule | Action |
|---|---|---|
| `decay_facts`           | Hourly  | Recompute `decay_score = exp(-Δt / half_life(fact_type))`. Allergy half-life = ∞. |
| `expire_facts`          | Hourly  | `UPDATE clinical_fact SET status='resolved' WHERE expires_at < now() AND status='active'`. |
| `consolidate_patient`   | On-write + daily | Recompute `patient_state.snapshot` from active facts. |
| `regenerate_summary`    | On-write if Δ above threshold | Re-prompt summarization LLM. |
| `prune_semantic_memory` | Weekly  | Drop snippets where `decay_score < 0.1 AND access_count = 0 AND age > 90d`. |
| `vacuum_audit`          | Monthly | Archive `conversation_event` rows older than retention window to cold storage. |

Recommended infra: `arq` (Redis-backed) or `Celery` with a beat scheduler. For Phase C we ship in-process `asyncio.create_task`.

---

## 13. Tradeoffs & Scaling

| Tradeoff | Discussion |
|---|---|
| **3 stores instead of 1** | More moving parts (Postgres, pgvector, Redis). Operationally heavier; mitigated by Docker compose for dev and one managed Postgres instance with `pgvector` for prod. |
| **LLM cost per turn** | Extraction adds one LLM call per user turn (~$0.0005 with a small model). Offset by saving full-context replay in the answer prompt. Net: roughly cost-neutral, latency +200–400ms before answer-LLM call. |
| **Eventual consistency** | Cache TTL = 5 minutes. After a write, the snapshot can lag for at most cache TTL. Acceptable for medical Q&A; not acceptable for, say, infusion pump control. |
| **Provenance overhead** | `supersedes_id` / `contradicts_id` chains can grow long for chronic patients. Mitigation: index the heads; UI/queries usually want only `status='active'`. |
| **pgvector scale** | At ~1M rows per node, HNSW recall stays >95% at ef_search=80. Beyond that, partition by `patient_id` hash. For most deployments, semantic_memory stays small because we don't embed everything. |
| **Write amplification** | Every turn writes: 1 ConversationEvent + N facts + 1 PatientState upsert + cache invalidate. Bounded; insert-heavy workload, Postgres handles tens of thousands per second on modest hardware. |
| **Async-only services** | Cleaner for FastAPI but the existing CLI uses `asyncio.run()` per call. Acceptable; new adapter exposes both sync + async surfaces. |

---

## 14. How This Improves Medical Safety

1. **Allergy collisions caught before LLM call.** `SafetyService.check_allergy(candidate_med, patient_id)` runs in `ExtractionService` before any drug suggestion lands in a prompt.
2. **Stale medications expire automatically.** No more "patient is on warfarin" inferred from a six-month-old turn when the patient stopped two months ago.
3. **Contradictions visible, not blended.** A new "no chest pain" turn marks the prior "chest pain" fact as `contradicted` with full provenance — the LLM sees only the current truth.
4. **Audit trail per answer.** `retrieval_log` records exactly which `clinical_fact` IDs entered the prompt for each `request_id`. Answering "why did the assistant say X?" becomes a single SQL query.
5. **Importance prioritization for safety-critical facts.** Allergies, current meds, and known interactions have `importance = 1.0` and `decay_score = 1.0` (no decay). They are *never* trimmed by the token-budget gate.
6. **No prompt drift from old conversations.** Three-week-old chitchat cannot poison today's clinical reasoning — it does not exist in the prompt.
7. **Identity-anchored memory.** Memory is keyed by `patient_id`, not `session_id`. A patient who comes back tomorrow has yesterday's facts available.

---

## 15. Open Questions / Future Work

- **Multi-clinician annotations.** Add a `clinician_note` table for free-form annotations distinct from extracted facts.
- **Patient-facing memory review.** UI to let patients see + correct their facts (HIPAA: full audit needed).
- **Federated extraction.** Currently one LLM extracts; for higher quality, run two and reconcile.
- **Structured ingest from HL7 FHIR.** Replace `source = 'patient_report'` with FHIR-pulled facts for hospital-integrated deployments.
- **Differential privacy on aggregate queries.** When/if we run cohort analytics over `clinical_fact`.

---

## 16. Filesystem Diff Summary

**Added (this redesign):**
- `docs/MEMORY_REDESIGN.md` (this document)
- `memory/` (new package — see §5)

**Modified:**
- `graphrag/pipeline/graphrag_pipeline.py` — Stage -2 calls `LongitudinalMemoryAdapter.load(patient_id)` instead of session-keyed adapter
- `graphrag/config/settings.py` — adds `DATABASE_URL`, `MEMORY_CACHE_TTL_SEC`, `EXTRACTION_MODEL`

**Removed (after Phase E):**
- `Memory_Layer/` (entire package)
- `graphrag/memory/session_adapter.py`

**Untouched:**
- All ingestion (`chunking/`, `ingest_*.py`) — operates on documents, not patient memory
- Retrievers (`graphrag/retrievers/`), gatekeeper, classifier, routing — orthogonal
