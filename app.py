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

def main():
    # Initialize environments and keys
    load_dotenv()
    init_telemetry()
    
    print("\n--- Agentic RAG Pipeline Verification ---")
    print(f"OpenAI Guard:    {'ACTIVE' if os.getenv('OPENAI_API_KEY') else 'MISSING'}")
    print(f"Pinecone Guard:  {'ACTIVE' if os.getenv('PINECONE_API_KEY') else 'MISSING'}")
    print(f"LangSmith Guard: {'ACTIVE' if os.getenv('LANGCHAIN_API_KEY') else 'MISSING'}")
    print("-------------------------------------------\n")

    # Extract the physical layout structure
    pdf_path = "data/Apple_10-Q.pdf"
    print(f"Extracting structural canvas from: {pdf_path}...")
    fingerprint = extract_structural_fingerprint(pdf_path)
    
    # Let the Ingestion Agent process the strategy call
    print("Invoking Ingestion Agent for architectural review...")
    strategy = determine_document_strategy(fingerprint)

    # Get the document chunks based on the agent's recommended strategy
    print("Slicing document using hybrid layout parser...")
    processed_chunks = chunk_document_hybrid(
        pdf_path = pdf_path,
        chunk_size = strategy.chunk_size,
        chunk_overlap = strategy.chunk_overlap
    )
    
    # Review the structured machine contract
    print("\n================= INGESTION STRATEGY CONTRACT =================")
    print(f"Document Classification: {strategy.document_type}")
    print(f"Assigned Strategy Tier:  {strategy.chunking_strategy}")
    print(f"Target Chunk Size:       {strategy.chunk_size} characters")
    print(f"Target Overlap Bounds:   {strategy.chunk_overlap} characters")
    print(f"\nArchitect Reasoning:   {strategy.reasoning}")
    print("\n================================================================\n")

# 4. Show a sample of the actual chunks ready for Pinecone
    print(f"\nSuccessfully generated {len(processed_chunks)} hybrid chunks!")
    
    # Let's print a diagnostic summary of the chunk sizes
    chunk_lengths = [c["metadata"]["chunk_length"] for c in processed_chunks]
    avg_length = sum(chunk_lengths) / len(chunk_lengths)
    print(f"--- Chunk Diagnostics ---")
    print(f"Average Chunk Size: {avg_length:.1f} characters")
    print(f"Max Chunk Size:     {max(chunk_lengths)} characters")
    print(f"Min Chunk Size:     {min(chunk_lengths)} characters")
    print(f"-------------------------\n")
    
    # Look for a dense chunk to preview
    dense_chunk = None
    for idx, chunk in enumerate(processed_chunks):
        if chunk["metadata"]["chunk_length"] > 800:
            dense_chunk = chunk
            dense_idx = idx
            break

    if dense_chunk:
        print(f"================= DENSE VECTOR CHUNK PREVIEW (INDEX #{dense_idx}) =================")
        print(f"Metadata Source:  {dense_chunk['metadata']['source']}")
        print(f"Metadata Page:    {dense_chunk['metadata']['page']}")
        print(f"Metadata Section: {dense_chunk['metadata']['section']}")
        print(f"Character Count:  {dense_chunk['metadata']['chunk_length']}")
        print(f"\nChunk Text Body:\n{dense_chunk['text']}")
        print("====================================================================\n")
    else:
        print("No dense chunks found matching criteria.")

if __name__ == "__main__":
    main()