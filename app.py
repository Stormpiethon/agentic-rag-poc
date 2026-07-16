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
from runtime.graph import compiled_graph

# Entry point for the application that calls stateful LangGraph workflow
def run_agentic_rag(user_question: str):
    print("\n" + "="*60)
    print(f" !!! INITIALIZING AGENTIC RAG SYSTEM")
    print(f"User Question: '{user_question}'")
    print("="*60)

    # Initialize the default state structure
    initial_state = {
        "question": user_question,
        "current_query": "",
        "query_history": [],
        "documents": [],
        "loop_step": 0,
        "final_response": ""
    }

    # Run the compiled state machine
    final_state = compiled_graph.invoke(initial_state)

    print("\n" + "="*60)
    print(" !!! FINAL AGENT RESPONSE")
    print("="*60)
    print(final_state["final_response"])
    print("="*60 + "\n")

def main():
    load_dotenv()
    init_telemetry()
    
    # Query 1: Answering with highly specific data present in the Apple 10-Q
    test_question = "What were Apple's Products and Services net sales for the three months ended December 27, 2025?"
    run_agentic_rag(test_question)

if __name__ == "__main__":
    main()