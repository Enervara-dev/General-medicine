from neo4j import GraphDatabase
from graphrag.config.settings import Config
from graphrag.utils.logger import get_logger

logger = get_logger(__name__)


class Neo4jRetriever:
    def __init__(self):
        if not Config.NEO4J_URI or not Config.NEO4J_USER or not Config.NEO4J_PWD:
            raise ValueError("Neo4j configurations are missing.")
        # Aura terminates idle connections after ~30 min and routes over TLS
        # (neo4j+s:// scheme). The driver picks up the scheme from the URI, so
        # local `bolt://` deployments keep working — these params are safe for
        # both. `max_connection_lifetime` < Aura's idle-kill prevents stale-
        # connection errors after the service has been quiet for a while.
        self.driver = GraphDatabase.driver(
            Config.NEO4J_URI,
            auth=(Config.NEO4J_USER, Config.NEO4J_PWD),
            max_connection_lifetime=30 * 60,   # 30 min — under Aura idle kill
            connection_timeout=15.0,
            keep_alive=True,
        )

    def retrieve_relations(self, entities: list, hops: int = 1, limit: int = 20) -> list:
        """
        Traverse the knowledge graph for `entities`.

        hops=1 → direct (A)-[r]-(B)
        hops=2 → indirect (A)-[r1]-(M)-[r2]-(B)  — used for drug interactions
        """
        logger.info(
            f"🕸️  [2/3] Graph traversal  →  hops: {hops}  |  "
            f"entities: {len(entities)}"
        )

        graph_context: list[str] = []
        if not entities:
            logger.info("⚠️  No entities to query graph with.")
            return graph_context

        try:
            with self.driver.session() as session:
                if hops == 1:
                    cypher = """
                        MATCH (e:Entity)-[r]-(x:Entity)
                        WHERE toLower(e.name) IN $entities
                        RETURN e.name AS src, type(r) AS rel, x.name AS tgt
                        LIMIT $limit
                    """
                    results = session.run(cypher, entities=entities, limit=limit)
                    for rec in results:
                        graph_context.append(
                            f"{rec['src']} -[{rec['rel']}]→ {rec['tgt']}"
                        )

                elif hops == 2:
                    cypher = """
                        MATCH (e:Entity)-[r1]-(m:Entity)-[r2]-(x:Entity)
                        WHERE toLower(e.name) IN $entities
                        RETURN e.name AS src,
                               type(r1) AS rel1, m.name AS mid,
                               type(r2) AS rel2, x.name AS tgt
                        LIMIT $limit
                    """
                    results = session.run(cypher, entities=entities, limit=limit)
                    for rec in results:
                        graph_context.append(
                            f"{rec['src']} -[{rec['rel1']}]→ {rec['mid']} "
                            f"-[{rec['rel2']}]→ {rec['tgt']}"
                        )

        except Exception as e:
            logger.error(f"❌ Neo4j failed: {e}")

        logger.info(f"✅ {len(graph_context)} graph relations retrieved.")
        return graph_context

    # ------------------------------------------------------------------
    # Legacy alias kept for backward compatibility
    # ------------------------------------------------------------------
    def retrieve_1hop_relations(self, entities: list, limit: int = 20) -> list:
        return self.retrieve_relations(entities, hops=1, limit=limit)

    def close(self):
        if self.driver:
            self.driver.close()
