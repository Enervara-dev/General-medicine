"""
Neo4j must be fully optional: disabled (the default, GRAPH_RETRIEVAL_ENABLED
= false) means no connection attempt and no required credentials; enabled
means the existing Neo4jRetriever is callable with no architecture change
(AUDIT_REPORT.md §4/§10/§12).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from graphrag.config.settings import Settings
from graphrag.retrievers.neo4j_retriever import (
    Neo4jRetriever,
    NullNeo4jRetriever,
    build_neo4j_retriever,
)

# ---------------------------------------------------------------------------
# Disabled mode (default)
# ---------------------------------------------------------------------------

def test_disabled_returns_null_retriever_without_touching_neo4j():
    with patch("graphrag.retrievers.neo4j_retriever.GraphDatabase") as mock_driver_cls:
        retriever = build_neo4j_retriever(False)
        mock_driver_cls.driver.assert_not_called()
    assert isinstance(retriever, NullNeo4jRetriever)


def test_null_retriever_has_no_driver():
    retriever = build_neo4j_retriever(False)
    assert retriever.driver is None


def test_null_retriever_retrieve_relations_returns_empty_list():
    retriever = build_neo4j_retriever(False)
    assert retriever.retrieve_relations(["fever", "cough"], hops=1) == []
    assert retriever.retrieve_relations(["fever"], hops=2, limit=5) == []
    assert retriever.retrieve_1hop_relations(["fever"]) == []


def test_null_retriever_close_does_not_raise():
    build_neo4j_retriever(False).close()  # must not raise, no driver to close


def test_null_retriever_accepts_specialty_kwarg_and_ignores_it():
    from app.specialty import build_default_registry

    retriever = build_neo4j_retriever(False)
    specialty = build_default_registry().get("cardiology")
    assert retriever.retrieve_relations(["x"], specialty=specialty) == []


def test_graph_retrieval_disabled_by_default_in_settings():
    assert Settings.model_fields["GRAPH_RETRIEVAL_ENABLED"].default is False


def test_validate_required_does_not_need_neo4j_password_when_disabled():
    s = Settings(
        PINECONE_API_KEY="x", GEMINI_API_KEY="y",
        NEO4J_PASSWORD=None, GRAPH_RETRIEVAL_ENABLED=False,
    )
    s.validate_required("api")  # must not raise


# ---------------------------------------------------------------------------
# Enabled mode (mocked — no real Neo4j required)
# ---------------------------------------------------------------------------

def test_validate_required_needs_neo4j_password_when_enabled():
    from graphrag.config.settings import ConfigError

    s = Settings(
        PINECONE_API_KEY="x", GEMINI_API_KEY="y",
        NEO4J_PASSWORD=None, GRAPH_RETRIEVAL_ENABLED=True,
    )
    with pytest.raises(ConfigError, match="NEO4J_PASSWORD"):
        s.validate_required("api")


def test_enabled_constructs_real_retriever_with_mocked_driver():
    with patch("graphrag.retrievers.neo4j_retriever.Config") as mock_config, \
         patch("graphrag.retrievers.neo4j_retriever.GraphDatabase") as mock_gdb:
        mock_config.NEO4J_URI = "bolt://localhost:7687"
        mock_config.NEO4J_USER = "neo4j"
        mock_config.NEO4J_PWD = "secret"
        mock_gdb.driver.return_value = MagicMock()

        retriever = build_neo4j_retriever(True)

        assert isinstance(retriever, Neo4jRetriever)
        mock_gdb.driver.assert_called_once()
        call_kwargs = mock_gdb.driver.call_args
        assert call_kwargs[0][0] == "bolt://localhost:7687"
        assert call_kwargs[1]["auth"] == ("neo4j", "secret")


def test_enabled_retrieve_relations_uses_mocked_session_1hop():
    with patch("graphrag.retrievers.neo4j_retriever.Config") as mock_config, \
         patch("graphrag.retrievers.neo4j_retriever.GraphDatabase") as mock_gdb:
        mock_config.NEO4J_URI = "bolt://localhost:7687"
        mock_config.NEO4J_USER = "neo4j"
        mock_config.NEO4J_PWD = "secret"

        mock_session = MagicMock()
        mock_session.run.return_value = [
            {"src": "chest pain", "rel": "CAUSES", "tgt": "myocardial infarction"},
        ]
        mock_driver = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_gdb.driver.return_value = mock_driver

        retriever = build_neo4j_retriever(True)
        result = retriever.retrieve_relations(["chest pain"], hops=1, limit=20)

        assert result == ["chest pain -[CAUSES]→ myocardial infarction"]
        mock_session.run.assert_called_once()


def test_enabled_retrieve_relations_empty_entities_short_circuits():
    with patch("graphrag.retrievers.neo4j_retriever.Config") as mock_config, \
         patch("graphrag.retrievers.neo4j_retriever.GraphDatabase") as mock_gdb:
        mock_config.NEO4J_URI = "bolt://localhost:7687"
        mock_config.NEO4J_USER = "neo4j"
        mock_config.NEO4J_PWD = "secret"
        mock_gdb.driver.return_value = MagicMock()

        retriever = build_neo4j_retriever(True)
        assert retriever.retrieve_relations([], hops=1) == []


def test_enabled_without_credentials_raises():
    with patch("graphrag.retrievers.neo4j_retriever.Config") as mock_config:
        mock_config.NEO4J_URI = "bolt://localhost:7687"
        mock_config.NEO4J_USER = "neo4j"
        mock_config.NEO4J_PWD = None
        with pytest.raises(ValueError, match="Neo4j configurations are missing"):
            build_neo4j_retriever(True)
