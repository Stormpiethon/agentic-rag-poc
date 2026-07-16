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
from utils.telemetry import init_telemetry, log_run_to_csv
from runtime.graph import compiled_graph


# Run the agentic RAG workflow with a user question
def run_agentic_rag(user_question: str):
    print("\n" + "="*60)
    print(f"🚀 INITIALIZING AGENTIC RAG SYSTEM")
    print(f"User Question: '{user_question}'")
    print("="*60)

    initial_state = {
        "question": user_question,
        "current_query": "",
        "query_history": [],
        "documents": [],
        "loop_step": 0,
        "final_response": "",
        "node_telemetry": []
    }

    # Run state machine
    final_state = compiled_graph.invoke(initial_state)

    print("\n" + "="*60)
    print("FINAL AGENT RESPONSE")
    print("="*60)
    print(final_state["final_response"])
    print("="*60)

    # Log individual node stats to our CSV!
    log_run_to_csv(
        question = user_question,
        final_response = final_state["final_response"],
        query_history = final_state.get("query_history", []),
        node_telemetry = final_state["node_telemetry"],
        is_successful = final_state.get("is_successful", True)
    )

def main():
    load_dotenv()
    init_telemetry()
    
    # Run test question
    test_question = "What were Apple's Products and Services net sales for the three months ended December 27, 2025?"
    run_agentic_rag(test_question)

if __name__ == "__main__":
    main()