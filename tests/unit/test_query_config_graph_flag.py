"""
QueryConfig no longer carries a dead per-query-type `graph_enabled` field
(AUDIT_REPORT.md §4/§8: it was defined but never read anywhere in the
pipeline — only the global Settings.GRAPH_RETRIEVAL_ENABLED switch and
`graph_hops > 0` ever gated the Neo4j call). This pins the field's removal
and that `graph_hops` remains the one still-honored per-query-type signal.
"""

from __future__ import annotations

import dataclasses

from graphrag.query_understanding.query_config import QUERY_CONFIGS, QueryConfig, get_config
from graphrag.query_understanding.query_types import QueryType


def test_query_config_has_no_graph_enabled_field():
    field_names = {f.name for f in dataclasses.fields(QueryConfig)}
    assert "graph_enabled" not in field_names
    assert "graph_hops" in field_names


def test_every_registered_query_type_resolves():
    for qt in QueryType:
        cfg = get_config(qt)
        assert isinstance(cfg, QueryConfig)


def test_out_of_context_skips_retrieval_entirely():
    cfg = QUERY_CONFIGS[QueryType.OUT_OF_CONTEXT]
    assert cfg.vector_top_k == 0
    assert cfg.graph_hops == 0


def test_unknown_query_type_falls_back_to_unknown_config():
    class _NotARealQueryType:
        pass

    cfg = get_config(_NotARealQueryType())  # type: ignore[arg-type]
    assert cfg is QUERY_CONFIGS[QueryType.UNKNOWN]
