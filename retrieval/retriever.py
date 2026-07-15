from typing import List, Dict, Any, Optional
from openai import OpenAI
from pinecone import Pinecone
from ingestion.vector_store import INDEX_NAME, EMBEDDING_MODEL

def query_vector_store(query_text: str, top_k: int = 5, section_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    # Initialize our API clients
    client = OpenAI()
    pc = Pinecone()
    index = pc.Index(INDEX_NAME)
    
    # Generate the mathematical vector for the user's query
    print(f"Embedding query: '{query_text}'...")
    response = client.embeddings.create(
        input=[query_text],
        model=EMBEDDING_MODEL
    )
    query_vector = response.data[0].embedding
    
    # Construct the metadata filter if specified - This allows us to target specific sections directly
    filter_dict = {}
    if section_filter:
        filter_dict["section"] = section_filter
        print(f"Applying metadata filter: section == '{section_filter}'")
        
    # Query Pinecone
    print(f"Searching Pinecone for top {top_k} matches...")
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        filter=filter_dict if filter_dict else None,
        include_metadata=True
    )
    
    # Format the search results into a clean list of matches
    matched_chunks = []
    for match in results.matches:
        matched_chunks.append({
            "score": match.score,
            "text": match.metadata["text"],
            "metadata": {
                "source": match.metadata["source"],
                "page": int(match.metadata["page"]),
                "section": match.metadata["section"]
            }
        })
        
    return matched_chunks