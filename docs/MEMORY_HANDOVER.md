# Memory Subsystem — Engineer Handover

**Audience:** the engineer owning the memory rebuild.
**Goal:** unify memory into the architecture in the handover deck (Working +
Demographic + Semantic + Episodic, behind a Memory Orchestrator) **without
breaking the live answer pipeline.**

Read this first, then [`MEMORY_REDESIGN.md`](MEMORY_REDESIGN.md) for the deep
schema/design (it predates you but describes ~80% of your plan and is already
implemented as dead code in `memory/`).

---

## 1. TL;DR — where things stand

- There are **three** memory systems today. Two are live, one is a built-but-
  unwired scaffold that closely matches your plan.
- Your job is mostly **wiring + unifying + adding first-class Demographic
  memory + a real Orchestrator** — not green-field.
- The live answer pipeline must keep working throughout. Section 4 lists the
  exact seams you must preserve.

| Your plan (deck) | Exists today as | Status |
|---|---|---|
| Working Memory | `Memory_Layer/session_memory/` (Redis, keyed by `session_id`) | ✅ live |
| Demographic Memory | `StructuredState.demographics` — a dict field on session state | ⚠️ not first-class |
| Semantic Memory (patient clinical facts) | `memory/` package: `clinical_fact` + `semantic_memory` (Postgres + pgvector) | ❌ **dead scaffold — your starting point** |
| Episodic Memory | `episodic/` (Pinecone, keyed by `user_id`) | ✅ live (only when `user_id` is passed) |
| Memory Orchestrator (Plan→Retrieve→Rank→Fuse→Govern) | inline in `AsyncOrchestrator`; governance scattered across `episodic/{contradiction,ranker,compression}` and `memory/consolidation` | partial / scattered |
| Clinical RAG | `graphrag/` (Pinecone vectors + Neo4j graph) | ✅ live, orthogonal — leave it |

> ⚠️ Naming trap: in the deck "Semantic Memory" = the **patient's** clinical
> facts (conditions/allergies/meds). In this repo "semantic/vector retrieval"
> usually means the **document** knowledge base (`graphrag/`). They are
> different things. Patient facts → `memory/clinical_fact`. Don't conflate.

---

## 2. The three systems in detail

### A. Working/session memory — `Memory_Layer/session_memory/` ✅ LIVE
Redis-backed (2 h TTL, in-memory fallback), keyed by `session_id`.
- Unit of memory = raw turns + rolling prose summary + `StructuredState`
  (regex-extracted symptoms/drugs/allergies/conditions/demographics).
- Reached through **two near-duplicate adapters**:
  - `app/services/memory/session.py` (async functions) → FastAPI path ✅
  - `graphrag/memory/session_adapter.py` (sync class, wraps `asyncio.run()`) →
    legacy CLI path. **Consolidate these into one async module.**

### B. Episodic memory — `episodic/` ✅ LIVE (optional)
Pinecone, namespaced by `user_id`. Strong pipeline already:
extract → contradiction-check → clarify → compress → store; retrieval does
vector + composite rerank (similarity × recency × priority × decay).
Wired at `app/services/orchestration/pipeline.py` Stage 3.5 (read) and Stage 5
(fire-and-forget ingest). Your deck's "Governance Layer" largely exists here.

### C. Longitudinal/clinical-fact memory — `memory/` ❌ DEAD SCAFFOLD (keep)
Postgres + pgvector, keyed by `patient_id`. **Imported nowhere** — zero rows
ever written. But it is fully built: ORM models, 2 Alembic migrations,
`ExtractionService`, `ConsolidationService` (merge/supersede/contradict),
`SafetyService` (allergy/interaction gates), `RetrievalService`,
`LongitudinalMemoryAdapter`. This is your scaffold — extend/wire it, don't
rebuild. Entry point: `memory/adapter.py::LongitudinalMemoryAdapter`.

---

## 3. Decisions to make in week 1 (these shape everything)

1. **Identity key.** Today: session=`session_id`, episodic=`user_id`,
   longitudinal=`patient_id`. Memory dies when a session TTLs out. **Recommend:
   make `patient_id` the spine** and key all durable memory on it; `session_id`
   becomes a short-lived working-memory scope only.
2. **Episodic store.** Deck says "Postgres-first episodic, Neo4j later." Repo
   uses Pinecone today (working) and Neo4j already exists for the KG. Decide:
   keep Pinecone episodic, or migrate to the Postgres `episodic_memory` table in
   `memory/`. Recommend: keep Pinecone short-term, revisit after the core lands.
3. **Orchestrator.** Deck shows a dedicated Memory Orchestrator with
   parallel retrieval + a governance layer. Today that logic is inline in
   `AsyncOrchestrator`. Decide whether to extract a `MemoryOrchestrator` service
   (recommended) and whether to adopt LangGraph (deck mentions it) or keep plain
   async — the repo has no LangGraph today; adding it is a real dependency call.
4. **Extraction unification.** There are currently **two** LLM/regex extractors
   on the same utterance (session regex + episodic LLM), and the dead `memory/`
   adds a third. Collapse to **one** extraction call whose output fans out to
   facts + episodes + demographics.

---

## 4. Seams you MUST preserve (don't break the live pipeline)

The answer pipeline (`app/services/orchestration/pipeline.py`) consumes memory
through these contracts. Keep the shapes; swap the implementation behind them.

- **Stage -2 load** — `load_session(mgr, session_id) -> SessionBundle` with
  `.session` and `.working_memory`; `build_retrieval_query(query, wm) -> str`.
- **Context assembly** — `assemble_memory_payload(...) -> ContextPayload` with
  `.memory_context` (str) and `.conversation_context` (str). These strings are
  injected verbatim into the answer prompt.
- **Demographic contract (subtle!)** — `_compose_answer_prompts` sets
  `has_name = "Patient name:" in memory_context`. The prompt's greet-by-name
  behavior depends on the memory block literally containing a `Patient name:`
  line. If you restructure the memory block, preserve this token (or update
  `prompt_layers.py` + its tests together).
- **Stage 3.5 episodic** — `_load_episodic_context(user_id, query) -> str`,
  prepended to `memory_context`.
- **Stage 5 save** — `save_after_turn(...)` and fire-and-forget
  `_ingest_episodic_safe(...)`.
- **Confidence stopping (just shipped on `feature/confidence-based-stopping`)** —
  the gatekeeper emits `diagnostic_confidence` + `leading_diagnosis`;
  `graphrag/domain/messages.py::is_terminal_turn` consumes them. If you rework
  the analyzer or memory, **keep these fields flowing** or follow-up-stopping
  regresses. See `tests/unit/test_terminal_turn.py`.

Both pipelines exist: the async FastAPI `AsyncOrchestrator` (production) and the
legacy sync `GraphRAGPipeline` (CLI). **Target the async path.** Either update
the CLI to the same async adapter or retire it — don't maintain two memory
implementations.

---

## 5. Suggested PR sequence (small, reviewable, always-green)

1. **Cleanup PR** — merge the two session adapters into one async module;
   unify `SessionBundle`/`MemoryAwareSession`; make the CLI call it. No behavior
   change. (This is the "Path 2" cleanup; do it first to de-risk.)
2. **Infra PR** — `DATABASE_URL` + async SQLAlchemy engine wired; run the two
   existing `memory/db/migrations`; add a `LONGITUDINAL_MEMORY_ENABLED` flag.
   Nothing reads from Postgres yet.
3. **Dual-write PR** — behind the flag, write `ConversationEvent` + extracted
   `ClinicalFact`s to Postgres alongside the live path. Verify rows, no reads.
4. **Demographic memory PR** — promote demographics to first-class facts;
   preserve the `Patient name:` prompt contract.
5. **Read-switch PR** — behind the flag, assemble `memory_context` from
   `LongitudinalMemoryAdapter` for a small % of traffic; compare answer quality.
6. **Orchestrator PR** — extract `MemoryOrchestrator` (parallel retrieve +
   governance) per the deck.
7. **Decommission PR** — once reads are on Postgres, delete `Memory_Layer/` and
   `graphrag/memory/session_adapter.py` (Phase E in `MEMORY_REDESIGN.md`).

Keep each PR flag-gated so it's reversible. `MEMORY_REDESIGN.md §11` has the
fuller migration phases.

---

## 6. Dev setup & guardrails

- **Run:** `uv sync` then `uv run pytest tests/unit -q` (120 tests, ~1s).
  Smoke: `uv run python smoke_test.py`.
- **Env:** copy `.env.example` → `.env`; needs `GEMINI_API_KEY`. Redis/Pinecone/
  Neo4j optional — code falls back (in-memory session, skipped episodic).
- **Don't** edit `app/services/orchestration/prompt_layers.py` rule text without
  updating `tests/unit/test_prompt_layers.py` — the prompt contract is
  test-locked (token budget + rule presence).
- **Add tests with every behavior change** — this repo treats memory as
  safety-critical (allergies must never be trimmed; contradictions resolved, not
  blended). See `MEMORY_REDESIGN.md §14`.
- The current feature work lives on branch `feature/confidence-based-stopping`
  (3 commits: deps, NDJSON validator fix, confidence stopping). Branch off
  `main` for memory work.

---

## 7. Open questions to resolve with the team

- HIPAA/GDPR (deck's cross-cutting layer): retention window for
  `conversation_event`, audit access, patient-facing fact correction.
- Background jobs (decay/expiry/consolidation): in-process `asyncio` to start,
  or `arq`/Celery? (`MEMORY_REDESIGN.md §12`.)
- Is `user_id` (episodic) the same identity as `patient_id` (longitudinal)? They
  must be reconciled before dual-write.
