import os
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
def initialize_and_upsert(chunks: List[Dict[str, Any]]):
    pc = Pinecone()

    # Check if index exists; if not, provision a new Serverless index
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    
    if INDEX_NAME not in existing_indexes:
        print(f"Index '{INDEX_NAME}' not found. Provisioning a new Serverless index...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric="cosine",
            # Change cloud and region as needed for deployment
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print("Index provisioned successfully!")
    else:
        print(f"Connected to existing Pinecone index: '{INDEX_NAME}'")

    # Connect to the target active index
    index = pc.Index(INDEX_NAME)

    # Batch process chunk embeddings to stay within rate bounds
    print(f"Generating embeddings for {len(chunks)} chunks...")
    texts_to_embed = [chunk["text"] for chunk in chunks]
    embeddings = generate_embeddings(texts_to_embed)

    # Format the final vectors payload for Pinecone upsert
    vectors_payload = []
    for idx, chunk in enumerate(chunks):
        # Generate a unique, deterministic ID for each vector slice
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

    # Execute the batch upsert (Chunking into groups of 100 to avoid payload size limits)
    batch_size = 100
    print("Upserting vector payloads to Pinecone...")
    for i in range(0, len(vectors_payload), batch_size):
        batch = vectors_payload[i:i + batch_size]
        index.upsert(vectors=batch)
        print(f"  Upserted batch {i // batch_size + 1}/{(len(vectors_payload) - 1) // batch_size + 1}...")

    print("Database upload completed successfully!")