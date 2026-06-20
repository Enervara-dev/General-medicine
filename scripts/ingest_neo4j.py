import os
import glob
import json
import logging
from tqdm import tqdm
from dotenv import load_dotenv
from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO, format='%(message)s')

VALID_RELATIONS = {
    "Disease": ["CAUSES", "ASSOCIATED_WITH", "INCREASES_RISK_OF"],
    "Drug": ["TREATS", "PREVENTS"],
    "Symptom": ["INDICATES"]
}

def normalize_type(entity_name, original_type):
    """Fallback typing for historically misclassified entities."""
    if not entity_name: 
        return original_type
        
    lower_name = entity_name.lower()
    mapping = {
        "muscle wasting": "Symptom",
        "disability": "Outcome",
        "acidotic breathing": "ClinicalSign"
    }
    
    if lower_name in mapping:
        return mapping[lower_name]
        
    if "breathing" in lower_name:
        return "ClinicalSign"
        
    return original_type

def normalize_relation(rel, context="strong"):
    """Normalize CAUSES based on context if available."""
    if rel == "CAUSES":
        if context == "weak":
            return "ASSOCIATED_WITH"
        elif context == "risk":
            return "INCREASES_RISK_OF"
    return rel

def validate_relation(src_type, rel):
    """Enforce strict ontology edges based on source node type."""
    allowed = VALID_RELATIONS.get(src_type)
    if allowed is not None:
        return rel in allowed
    return True # Allow other types not explicitly restricted

def load_chunks(base_dir=os.path.join("chunking", "output", "v1")):
    """Load all chunk JSON files."""
    logging.info(f"Loading chunks from {base_dir}...")
    chunk_files = glob.glob(os.path.join(base_dir, "**", "*.json"), recursive=True)
    chunks = []
    
    for file_path in chunk_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                chunks.append(json.load(f))
        except Exception:
            pass
            
    logging.info(f"Loaded {len(chunks)} chunks.")
    return chunks

class Neo4jIngester:
    def __init__(self):
        load_dotenv()
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USERNAME")
        pwd = os.getenv("NEO4J_PASSWORD")
        
        if not uri or not user or not pwd:
            raise ValueError("Neo4j Database Credentials missing from .env\nPlease add NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD.")
            
        logging.info("Connecting to Neo4j database...")
        # Same Aura-friendly knobs as the runtime retriever — long ingest runs
        # against Aura would otherwise drop idle connections mid-batch.
        self.driver = GraphDatabase.driver(
            uri,
            auth=(user, pwd),
            max_connection_lifetime=30 * 60,
            connection_timeout=15.0,
            keep_alive=True,
        )
        self.driver.verify_connectivity()
        self._create_constraint()

    def close(self):
        self.driver.close()

    def _create_constraint(self):
        """Ensure entity naming is strictly unique at the database level."""
        with self.driver.session() as session:
            session.run("CREATE CONSTRAINT entity_name_unique IF NOT EXISTS FOR (n:Entity) REQUIRE n.name IS UNIQUE")
            logging.info("✅ Global UNIQUE constraint ensured on :Entity(name).")

    def process_chunks(self, chunks):
        """Processes chunks safely with strict unique relationship MERGE limits."""
        # Threshold constants
        MAX_RELATIONS_PER_NODE = 20 
        
        with self.driver.session() as session:
            # We track relationships globally to artificially limit fan-out noise per source node
            src_relation_counts = {}
            
            for chunk in tqdm(chunks, desc="Ingesting to Neo4j"):
                entities = chunk.get("entities", [])
                relations = chunk.get("relations", [])
                
                # --- PROCESS NODES ---
                nodes_by_label = {}
                node_type_map = {} # Track types for relation validation
                
                for ent in entities:
                    ent_name = ent.get("normalized_name") or ent.get("name")
                    if not ent_name: 
                        continue
                        
                    ent_name = ent_name.strip()
                    original_type = ent.get("type", "Unknown")
                    
                    # 1. Normalize Entity Types
                    refined_type = normalize_type(ent_name, original_type)
                    dynamic_label = refined_type.replace(" ", "_").title()
                    
                    node_type_map[ent_name] = dynamic_label
                    
                    if dynamic_label not in nodes_by_label:
                         nodes_by_label[dynamic_label] = []
                    nodes_by_label[dynamic_label].append({"name": ent_name})
                    
                # Execute Node MERGE per label group safely
                for d_label, node_data in nodes_by_label.items():
                    cypher_nodes = f"""
                    UNWIND $nodes AS node
                    MERGE (n:Entity {{name: node.name}})
                    SET n:{d_label}
                    """
                    session.run(cypher_nodes, nodes=node_data)
                
                # --- PROCESS RELATIONSHIPS ---
                rels_by_type = {}
                for rel in relations:
                    src = rel.get("source")
                    dst = rel.get("target")
                    if not src or not dst:
                        continue
                        
                    src = src.strip()
                    dst = dst.strip()
                    raw_type = rel.get("type", "RELATED_TO").replace(" ", "_").upper()
                    
                    # 2. Normalize Relations safely (assume 'strong' unless chunk metadata says otherwise)
                    # We can pass context here if derived from text
                    rel_type = normalize_relation(raw_type, context="strong")
                    
                    # 3. Validate Constraints
                    src_label = node_type_map.get(src, "Unknown")
                    if not validate_relation(src_label, rel_type):
                        continue # Drop invalid relation edge
                        
                    # 4. Filter Graph Noise Limits
                    if src_relation_counts.get(src, 0) >= MAX_RELATIONS_PER_NODE:
                        continue # Drop highly fanned-out edges
                    
                    src_relation_counts[src] = src_relation_counts.get(src, 0) + 1
                        
                    if rel_type not in rels_by_type:
                        rels_by_type[rel_type] = []
                    rels_by_type[rel_type].append({"src": src, "dst": dst})
                        
                # Execute Relationship MERGE per relation type strictly with ON CREATE timestamps
                for r_type, rel_data in rels_by_type.items():
                    cypher_rels = f"""
                    UNWIND $rels AS rel
                    MATCH (s:Entity {{name: rel.src}})
                    MATCH (t:Entity {{name: rel.dst}})
                    MERGE (s)-[r:{r_type}]->(t)
                    ON CREATE SET r.created_at = timestamp()
                    """
                    session.run(cypher_rels, rels=rel_data)

    def test_queries(self):
        """Run quick verification queries to print data to the console."""
        with self.driver.session() as session:
            logging.info("\n" + "="*50)
            logging.info("🔍 Verification: TEST QUERY 1 (First 10 Nodes)")
            nodes = session.run("MATCH (n:Entity) RETURN n.name AS name, labels(n) AS labels LIMIT 10")
            for record in nodes:
                labels = [l for l in record['labels']]
                logging.info(f"   Node: '{record['name']}' | Labels: {labels}")

            logging.info("\n" + "="*50)
            logging.info("🔍 Verification: TEST QUERY 2 (First 10 Relationships)")
            rels = session.run("MATCH (s)-[r]->(t) RETURN s.name AS src, type(r) AS rel_type, t.name AS dst LIMIT 10")
            for record in rels:
                logging.info(f"   ({record['src']})  -[:{record['rel_type']}]->  ({record['dst']})")
                
            logging.info("="*50 + "\n")

if __name__ == "__main__":
    chunks = load_chunks()
    if not chunks:
         logging.warning("No chunks found!")
    else:
         ingester = Neo4jIngester()
         try:
             ingester.process_chunks(chunks)
             ingester.test_queries()
             logging.info("🎉 Neo4j Ingestion Completely Finished!")
         except Exception as e:
             logging.error(f"❌ Error during Graph ingestion: {e}")
         finally:
             ingester.close()
