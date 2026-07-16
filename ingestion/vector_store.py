import os
import json
from typing import List, Dict, Any
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

EMBEDDING_MODEL = "text-embedding-3-small"
INDEX_NAME = "financial-agent-rag"
DIMENSION = 1536

# OpenAI Embedding Client 
def generate_embeddings(texts: List[str]) -> List[List[float]]:
    client = OpenAI()
    response = client.embeddings.create(
        input = texts,
        model = EMBEDDING_MODEL
    )
    # Extract the vectors in the exact order they were requested
    return [item.embedding for item in response.data]


# Initialize Pinecone and upsert the vector payloads into the index
def initialize_and_upsert(chunks: List[Dict[str, Any]], strategy: Any):
    pc = Pinecone()

    existing_indexes = [idx.name for idx in pc.list_indexes()]
    
    # Verify if the index exists; if not, create a new Serverless index
    if INDEX_NAME not in existing_indexes:
        print(f"Index '{INDEX_NAME}' not found. Provisioning a new Serverless index...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print("Index provisioned successfully!")
    else:
        print(f"Connected to existing Pinecone index: '{INDEX_NAME}'")

    index = pc.Index(INDEX_NAME)

    # =============================================================================
    # IDEMPOTENCY STEP: Delete old document vectors before upserting
    if chunks:
        # Get the unique source path we are processing (e.g., 'data/Apple_10-Q.pdf')
        source_file = chunks[0]["metadata"]["source"]
        print(f"Purging existing vectors for: {source_file} to prevent duplicates...")
        
        # Pinecone allows us to delete vectors by matching a metadata filter
        index.delete(filter={"source": {"$eq": source_file}})
        print("Purge complete. Index is clean.")
    # =============================================================================

    # Generate embeddings for all chunks and prepare the upsert payload
    print(f"Generating embeddings for {len(chunks)} chunks...")
    texts_to_embed = [chunk["text"] for chunk in chunks]
    embeddings = generate_embeddings(texts_to_embed)

    vectors_payload = []
    for idx, chunk in enumerate(chunks):
        vector_id = f"{os.path.basename(chunk['metadata']['source'])}-p{chunk['metadata']['page']}-c{idx}"
        vectors_payload.append((
            vector_id,
            embeddings[idx],
            {
                "text": chunk["text"],
                "source": chunk["metadata"]["source"],
                "page": chunk["metadata"]["page"],
                "section": chunk["metadata"]["section"]
            }
        ))

    # Upsert the vectors in batches to Pinecone
    batch_size = 100
    print("Upserting vector payloads to Pinecone...")
    for i in range(0, len(vectors_payload), batch_size):
        batch = vectors_payload[i:i + batch_size]
        index.upsert(vectors=batch)
        print(f"  Upserted batch {i // batch_size + 1}/{(len(vectors_payload) - 1) // batch_size + 1}...")

    # Write the Strategy Contract to a local JSON trace file for future audit runs
    contract_path = os.path.join("ingestion", "strategy_contract.json")
    with open(contract_path, "w") as f:
        json.dump({
            "document_type": strategy.document_type,
            "chunking_strategy": strategy.chunking_strategy,
            "chunk_size": strategy.chunk_size,
            "chunk_overlap": strategy.chunk_overlap,
            "reasoning": strategy.reasoning
        }, f, indent=4)
        
    print("Database upload completed successfully!")