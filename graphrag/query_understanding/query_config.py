from dataclasses import dataclass

from graphrag.query_understanding.query_types import QueryType


@dataclass
class QueryConfig:
    """
    Drives ALL pipeline behaviour for a given query type.
    Every downstream component reads from this — no hardcoded logic elsewhere.

    NOTE: there is no per-query-type graph toggle here. Per AUDIT_REPORT.md
    §4/§8, a ``graph_enabled`` field used to live on this dataclass but was
    never read anywhere in the pipeline — only the global
    ``Settings.GRAPH_RETRIEVAL_ENABLED`` switch and ``graph_hops > 0`` ever
    gated the Neo4j call (app/services/orchestration/pipeline.py). That dead
    field has been removed rather than wired in, since wiring it in would be
    a behavior change and the task is to make the EXISTING global flag the
    single source of truth, not to add new graph-gating logic. Set
    ``graph_hops=0`` on a QueryConfig to skip graph traversal for that query
    type regardless of the global flag.
    """
    query_type:             QueryType
    vector_top_k:           int         # how many candidates to pull from Pinecone
    reranker_top_k:         int         # how many to keep after reranking
    graph_hops:             int         # 1 or 2-hop Neo4j traversal; 0 = skip graph for this query type
    priority_entity_types:  list[str]   # entity types to surface / boost
    goal:                   str         # human-readable description (logged)
    boost_drug_pairs:       bool = False  # special flag for drug_interaction only


# ---------------------------------------------------------------------------
# Registry — one config object per query type
# ---------------------------------------------------------------------------
QUERY_CONFIGS: dict[QueryType, QueryConfig] = {

    QueryType.SYMPTOM_QUERY: QueryConfig(
        query_type            = QueryType.SYMPTOM_QUERY,
        vector_top_k          = 15,      # oversample before reranker
        reranker_top_k        = 5,
        graph_hops            = 1,
        priority_entity_types = ["disease", "symptom", "syndrome"],
        goal                  = "cause identification",
    ),

    QueryType.DRUG_INTERACTION: QueryConfig(
        query_type            = QueryType.DRUG_INTERACTION,
        vector_top_k          = 15,      # wider net for drug combos
        reranker_top_k        = 5,
        graph_hops            = 2,       # 2-hop to find indirect interactions
        priority_entity_types = ["drug", "drug_class", "mechanism", "side_effect"],
        goal                  = "interaction and risk",
        boost_drug_pairs      = True,
    ),

    QueryType.DIAGNOSIS: QueryConfig(
        query_type            = QueryType.DIAGNOSIS,
        vector_top_k          = 15,
        reranker_top_k        = 5,
        graph_hops            = 1,
        priority_entity_types = ["disease", "syndrome", "condition", "disorder"],
        goal                  = "clear explanation / definition",
    ),

    QueryType.GUIDELINE: QueryConfig(
        query_type            = QueryType.GUIDELINE,
        vector_top_k          = 20,      # deepest retrieval
        reranker_top_k        = 7,       # keep more for structured protocols
        graph_hops            = 1,
        priority_entity_types = ["procedure", "drug", "treatment", "protocol", "therapy"],
        goal                  = "structured clinical protocol",
    ),

    QueryType.LAB_INTERPRETATION: QueryConfig(
        query_type            = QueryType.LAB_INTERPRETATION,
        vector_top_k          = 15,
        reranker_top_k        = 5,
        graph_hops            = 1,
        priority_entity_types = ["test", "lab_value", "biomarker", "threshold"],
        goal                  = "lab result interpretation",
    ),

    QueryType.PROGNOSIS: QueryConfig(
        query_type            = QueryType.PROGNOSIS,
        vector_top_k          = 15,
        reranker_top_k        = 5,
        graph_hops            = 1,
        priority_entity_types = ["outcome", "risk_factor", "survival", "mortality", "disease"],
        goal                  = "future risk / survival estimate",
    ),

    QueryType.OUT_OF_CONTEXT: QueryConfig(
        query_type            = QueryType.OUT_OF_CONTEXT,
        vector_top_k          = 0,
        reranker_top_k        = 0,
        graph_hops            = 0,
        priority_entity_types = [],
        goal                  = "reject out of context / gibberish queries",
    ),

    QueryType.UNKNOWN: QueryConfig(
        query_type            = QueryType.UNKNOWN,
        vector_top_k          = 15,
        reranker_top_k        = 5,
        graph_hops            = 1,
        priority_entity_types = [],
        goal                  = "general medical answer",
    ),
}


def get_config(query_type: QueryType) -> QueryConfig:
    """Retrieve the pipeline config for a classified query type."""
    return QUERY_CONFIGS.get(query_type, QUERY_CONFIGS[QueryType.UNKNOWN])
