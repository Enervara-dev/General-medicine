# syntax=docker/dockerfile:1.6
# ---------------------------------------------------------------------------
# Multi-stage build for the Enervera FastAPI service.
#   Stage 1 (builder): install Python deps into an isolated layer.
#   Stage 2 (runtime): copy the installed deps + source, run as non-root.
#
# Local:
#   docker build -t enervera-api .
#   docker run -p 8000:8000 --env-file .env enervera-api
#
# Render: uses this file directly (render.yaml runtime: docker).
# ---------------------------------------------------------------------------

ARG PYTHON_VERSION=3.11-slim


FROM python:${PYTHON_VERSION} AS builder

WORKDIR /app

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install only the metadata + minimal source needed for the editable install
# to resolve. The full source is copied in stage 2; this keeps the build cache
# valid across most edits.
COPY pyproject.toml README.md ./
# `main.py` is a force-include target in pyproject.toml (legacy CLI entry
# point). Hatchling validates it exists at editable-install time, even
# though the API path doesn't use it at runtime.
COPY main.py                     ./main.py
COPY graphrag/__init__.py        ./graphrag/__init__.py
COPY episodic/__init__.py        ./episodic/__init__.py
COPY chunking/__init__.py        ./chunking/__init__.py
COPY Memory_Layer/__init__.py    ./Memory_Layer/__init__.py
# scripts/ is a listed wheel package — Hatchling validates the dir exists at
# build time even for editable installs.
COPY scripts/__init__.py         ./scripts/__init__.py

RUN pip install --no-cache-dir -e ".[api]"


FROM python:${PYTHON_VERSION} AS runtime

WORKDIR /app

# Re-use the deps installed in the builder layer.
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the application source last so iterative code edits only invalidate
# this layer, not the (much larger) pip install layer above.
COPY . .

# Drop to non-root.
RUN useradd --create-home --uid 10001 enervera && chown -R enervera:enervera /app
USER enervera

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    LOG_LEVEL=INFO

EXPOSE 8000

# Render injects $PORT. Use a shell form so the env var is expanded at runtime.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips=*"]
