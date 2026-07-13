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

def main():
    # Load your private API keys from the .env file
    load_dotenv()
    
    # Kick off the tracing engine before doing anything else
    init_telemetry()
    
    print("\n--- Agentic RAG Environment Hook Check ---")
    print(f"OpenAI Key Configured: {'PINKEY_FOUND' if os.getenv('OPENAI_API_KEY') else 'MISSING'}")
    print(f"Pinecone Key Configured: {'PINKEY_FOUND' if os.getenv('PINECONE_API_KEY') else 'MISSING'}")
    print("-------------------------------------------\n")
    
    # LangGraph runtime workflow will trigger right here!

if __name__ == "__main__":
    main()