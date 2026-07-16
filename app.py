"""
Agentic RAG POC Application
Alexander Brittain
Tech Consulting LLC

Parse and chunk financial documents from SEC related to Microsoft and Apple to show agentic
application of RAG (Retrieval-Augmented Generation) for financial analysis and insights.

app.py is the entry point for the application. It will read your environment variables, 
initialize your tracing telemetry, and let you pass test financial questions straight 
into the compiled graph engine
"""

import os
from dotenv import load_dotenv
from utils.telemetry import init_telemetry
from ingestion.parsers import extract_structural_fingerprint, chunk_document_hybrid
from ingestion.agent import determine_document_strategy
from ingestion.vector_store import initialize_and_upsert
from retrieval.retriever import query_vector_store

def main():
    load_dotenv()
    init_telemetry()
    
    print("\n--- Running Full Pipeline (Ingestion + Trace Creation) ---")
    pdf_path = "data/Apple_10-Q.pdf"
    
    # Extract layout blueprint
    fingerprint = extract_structural_fingerprint(pdf_path)
    
    # Let the Agent set the rules
    strategy = determine_document_strategy(fingerprint)
    
    # Execute the chunker
    processed_chunks = chunk_document_hybrid(
        pdf_path=pdf_path,
        chunk_size=strategy.chunk_size,
        chunk_overlap=strategy.chunk_overlap
    )
    
    # Clean purge old vectors, upload new ones, and write the local strategy_contract.json
    initialize_and_upsert(processed_chunks, strategy)
    
    # Immediately test a retrieval query
    print("\n--- Running Retrieval Test ---")
    question = "What were Apple's Products and Services net sales for the three months ended December 27, 2025?"
    results = query_vector_store(query_text=question, top_k=1)
    
    if results:
        print(f"\nMatch Score: {results[0]['score']:.4f}")
        print(f"Content:\n{results[0]['text'][:400]}...")

if __name__ == "__main__":
    main()