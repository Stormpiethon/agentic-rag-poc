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
from ingestion.parsers import extract_structural_fingerprint
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
    
    # Review the structured machine contract
    print("\n================= INGESTION STRATEGY CONTRACT =================")
    print(f"Document Classification: {strategy.document_type}")
    print(f"Assigned Strategy Tier:  {strategy.chunking_strategy}")
    print(f"Target Chunk Size:       {strategy.chunk_size} characters")
    print(f"Target Overlap Bounds:   {strategy.chunk_overlap} characters")
    print(f"\nArchitect Reasoning:   {strategy.reasoning}")
    print("\n================================================================\n")

if __name__ == "__main__":
    main()