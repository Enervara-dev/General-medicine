# Installing pgvector for the longitudinal memory layer

`pgvector` is a Postgres extension that adds vector types and an HNSW index.
The core memory schema (migration `0001_initial`) does **not** require it —
only `semantic_memory` (migration `0002_semantic_memory_pgvector`) does.

If you tried `alembic upgrade head` and saw:

```
asyncpg.exceptions.FeatureNotSupportedError: extension "vector" is not available
DETAIL:  Could not open extension control file
  "C:/Program Files/PostgreSQL/17/share/extension/vector.control":
  No such file or directory.
```

…the extension binaries are missing from your Postgres server. Fix one of three ways.

---

## Option 1 — Use the official `pgvector/pgvector` Docker image (recommended)

Easiest, most reliable. Stops your local EDB Postgres 17 service first; runs Postgres 17 + pgvector in a container instead.

```powershell
# 1) Stop the EDB Postgres service so it doesn't fight for port 5432
Stop-Service postgresql-x64-17

# 2) Pull and run pgvector-enabled Postgres 17
docker run -d `
  --name enervera-pg `
  -e POSTGRES_PASSWORD=postgres `
  -e POSTGRES_DB=enervera `
  -p 5432:5432 `
  -v enervera-pgdata:/var/lib/postgresql/data `
  pgvector/pgvector:pg17

# 3) Verify
docker exec -it enervera-pg psql -U postgres -d enervera `
  -c "CREATE EXTENSION IF NOT EXISTS vector; SELECT extname FROM pg_extension;"
```

Your `DATABASE_URL` stays the same:

```
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/enervera
```

Then re-run:

```powershell
alembic upgrade head
```

Both migrations apply cleanly.

---

## Option 2 — Install pgvector into your existing EDB Postgres 17 (Windows)

pgvector does not ship pre-built Windows binaries from upstream. Two routes:

### 2a — Community pre-built DLL

1. Download a Windows build for Postgres 17 from a trusted community source (e.g. <https://github.com/pgvector/pgvector/issues/54> tracks Windows builds; verify against checksums).
2. Copy the files into your install:

   ```
   vector.dll                → C:\Program Files\PostgreSQL\17\lib\
   vector.control            → C:\Program Files\PostgreSQL\17\share\extension\
   vector--<version>.sql     → C:\Program Files\PostgreSQL\17\share\extension\
   ```

3. Restart the Postgres service:

   ```powershell
   Restart-Service postgresql-x64-17
   ```

4. Confirm:

   ```powershell
   & "C:\Program Files\PostgreSQL\17\bin\psql.exe" -U postgres -d enervera `
     -c "CREATE EXTENSION IF NOT EXISTS vector; \dx"
   ```

### 2b — Build from source (requires Visual Studio Build Tools)

```powershell
# Install Build Tools first (Visual Studio Installer → "Desktop development with C++")
cd C:\dev
git clone --branch v0.7.4 https://github.com/pgvector/pgvector.git
cd pgvector

# Use the x64 Native Tools Command Prompt:
set "PGROOT=C:\Program Files\PostgreSQL\17"
nmake /F Makefile.win
nmake /F Makefile.win install
```

Then restart the service and run `CREATE EXTENSION vector;`.

---

## Option 3 — Skip semantic memory for now

The longitudinal layer works without pgvector. Only the narrow vector-recall
fallback in `RetrievalService` becomes a no-op (which is fine — structured
retrieval handles most queries).

Run only migration `0001`:

```powershell
alembic upgrade 0001_initial
```

When you later install pgvector, finish with:

```powershell
alembic upgrade head     # applies 0002_semantic_memory_pgvector
```

`RetrievalService` checks for `semantic_memory` at query time; if the table
isn't there, semantic recall is silently skipped. No code changes needed.

---

## Verifying the install (any option)

```powershell
& "C:\Program Files\PostgreSQL\17\bin\psql.exe" -U postgres -d enervera -c "
SELECT extname, extversion FROM pg_extension WHERE extname IN ('pgcrypto','vector');
SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename;
"
```

Expected output after both migrations:

```
  extname  | extversion
-----------+------------
 pgcrypto  | 1.3
 vector    | 0.7.4

      tablename
----------------------
 alembic_version
 clinical_fact
 conversation_event
 episodic_memory
 patient
 patient_state
 retrieval_log
 semantic_memory
```
