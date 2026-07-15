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
from retrieval.retriever import query_vector_store

def main():
    load_dotenv()
    init_telemetry()
    
    print("\n--- Running Retrieval Engine Verification ---")
    
    # Define a highly specific financial question about Apple's Q1 performance
    question = "What were Apple's Products and Services net sales for the three months ended December 27, 2025?"
    
    # Execute our search
    results = query_vector_store(query_text=question, top_k=2)
    
    # Display the matches
    print(f"\nSuccessfully retrieved {len(results)} matches!")
    
    for idx, match in enumerate(results):
        print(f"\n================= MATCH #{idx + 1} (Cosine Similarity: {match['score']:.4f}) =================")
        print(f"Source:  {match['metadata']['source']} (Page {match['metadata']['page']})")
        print(f"Section: {match['metadata']['section']}")
        print(f"----------------------------------------------------------------------")
        print(f"Text Content Snippet:\n{match['text'].strip()[:800]}") # Show up to 800 chars of the match
        print("========================================================================\n")

if __name__ == "__main__":
    main()