"""
Shared pytest fixtures for the Enervera test suite.

Integration tests hit real Pinecone + Gemini, so we pin a single event loop
for the whole session — the genai SDK's async http client caches itself
against the first loop it sees, and pytest-asyncio's per-function loops
would cause the same "Event loop is closed" failure the main pipeline hit.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

# Episodic package's __init__ does the Windows TLS bootstrap (truststore)
# before any HTTPS call. Importing it here ensures that runs before any
# fixture / test tries to reach Pinecone or Gemini.
import episodic  # noqa: F401


@pytest.fixture(scope="session")
def event_loop():
    """One event loop for the whole test session — mirrors GraphRAGPipeline."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def episodic_container():
    """
    Build the episodic services container once and reuse across all tests.

    Tests assume the three personas (cardio_test, migraine_test, geriatric_test)
    are already seeded — run `python -m episodic.scripts.seed --all` first.
    """
    from episodic.api.dependencies import build_container

    container = build_container()
    return container


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "episodic" / "fixtures"
