import os
import glob
import json
import logging
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_chunks(base_dir="chunking/output/v1"):
    """Load all chunk JSON files from the specified directory."""
    logging.info(f"Loading chunks from {base_dir}...")
    chunk_files = glob.glob(os.path.join(base_dir, "**", "*.json"), recursive=True)
    chunks = []
    
    for file_path in chunk_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                chunks.append(data)
        except Exception as e:
            logging.error(f"Error loading {file_path}: {e}")
            
    logging.info(f"Loaded {len(chunks)} chunks.")
    return chunks

def extract_metadata_and_text(chunk):
    """
    Given a chunk dict, format the embedding text and output flattened metadata.
    """
    chunk_id = chunk.get("chunk_id", "unknown")
    text = chunk.get("text", "")
    summary = chunk.get("summary", "")
    source = chunk.get("source", {})
    book = source.get("book", "unknown")
    topic = source.get("topic", "unknown")
    
    # Flatten entities
    raw_entities = chunk.get("entities", [])
    flattened_entities = []
    for ent in raw_entities:
        ent_type = ent.get("type", "unknown")
        # Ensure we don't crash if name is missing
        ent_name = ent.get("normalized_name", ent.get("name", "unknown"))
        flattened_entities.append(f"{ent_type}: {ent_name}")
        
    # Build Embedding Text: Summary + Flattened Entities string + Text
    entities_str = "\n".join(flattened_entities)
    embed_text = f"SUMMARY:\n{summary}\n\nENTITIES:\n{entities_str}\n\nTEXT:\n{text}"
    
    metadata = {
        "chunk_id": chunk_id,
        "entities": flattened_entities, # List of strings format for Pinecone
        "book": book,
        "topic": topic,
        "summary": summary[:300]
    }
    
    return f"{book}_{chunk_id}", embed_text, metadata

def process_and_upsert_batches():
    pinecone_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_key:
        raise ValueError("PINECONE_API_KEY not set in .env")

    # Initialize Pinecone Client
    pc = Pinecone(api_key=pinecone_key)
    
    index_name = "enervera"
    logging.info(f"Connecting to Pinecone index: {index_name}")
    index = pc.Index(index_name)

    chunks = load_chunks()
    if not chunks:
        logging.warning("No chunks found to ingest. Exiting.")
        return

    # Use smaller batches to stay within Pinecone embedding limits (usually max 96 per request)
    batch_size = 50 
    model_name = "llama-text-embed-v2"
    
    # Optional parameters for inference depend on the model (like passage or query)
    embed_params = {"input_type": "passage", "truncate": "END"}

    for i in tqdm(range(0, len(chunks), batch_size), desc="Upserting Batches"):
        batch_chunks = chunks[i: i + batch_size]
        
        batch_ids = []
        batch_texts = []
        batch_metadata = []
        
        for c in batch_chunks:
            c_id, embed_text, metadata = extract_metadata_and_text(c)
            batch_ids.append(c_id)
            batch_texts.append(embed_text)
            batch_metadata.append(metadata)
            
        try:
            # 1. Generate Embeddings using Pinecone Inference API
            response = pc.inference.embed(
                model=model_name,
                inputs=batch_texts,
                parameters=embed_params
            )
            
            # Extract actual dense vectors
            embeddings = [emb["values"] for emb in response]

            # 2. Upsert Vectors
            vectors_to_upsert = []
            for j in range(len(batch_ids)):
                vectors_to_upsert.append(
                    {
                        "id": batch_ids[j], 
                        "values": embeddings[j], 
                        "metadata": batch_metadata[j]
                    }
                )
            
            index.upsert(vectors=vectors_to_upsert)

        except Exception as e:
            logging.error(f"Error on batch {i} to {i+batch_size}: {e}")
            break

def verify_with_query():
    """Verify integration by querying the index."""
    pinecone_key = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=pinecone_key)
    index = pc.Index("enervera")
    
    test_query = "What are the common symptoms of Myocarditis?"
    logging.info(f"\n--- Running Verification Query ---")
    logging.info(f"Query: {test_query}")
    
    try:
        # Embed query text (Pinecone limits prompt input_type='query' for some models, others take it raw)
        embed_params = {"input_type": "query", "truncate": "END"}
        response = pc.inference.embed(
            model="llama-text-embed-v2",
            inputs=[test_query],
            parameters=embed_params
        )
        query_vector = response[0]["values"]
        
        # Search Top K
        search_result = index.query(
            vector=query_vector,
            top_k=3,
            include_metadata=True
        )
        
        logging.info("Top 3 Matches:")
        for idx, match in enumerate(search_result.get("matches", [])):
             score = match.get("score")
             md = match.get("metadata", {})
             logging.info(f"\nResult {idx+1} [Score: {score:.4f}] - Chunk ID: {md.get('chunk_id')}")
             logging.info(f"Entities: {md.get('entities')}")
             logging.info(f"Summary Context: {md.get('summary')}")
             
             
    except Exception as e:
        logging.error(f"Query test failed: {e}")

if __name__ == "__main__":
    process_and_upsert_batches()
    verify_with_query()
    logging.info("Tasks Complete.")
