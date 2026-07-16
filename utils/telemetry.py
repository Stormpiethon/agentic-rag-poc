"""
Every time an agent makes a decision or queries pinecone, a detailed, step-by-step log of 
the decision-making process is generated. This log includes information about the agent's 
reasoning, the data it accessed, and the actions it took.
"""

import os
import csv
import json
from datetime import datetime


# Telemetry Initialization
def init_telemetry():
    # Ensure any necessary environment parameters or console styles are loaded
    print("✓ Telemetry systems initialized.")


# Telemetry Logging into a csv file for metrics analysis and performance tracking
def log_run_to_csv(question: str, final_response: str, query_history: list, node_telemetry: list, is_successful: bool, csv_path: str = "agent_telemetry.csv"):
    file_exists = os.path.exists(csv_path)
    
    headers = [
        "Timestamp", 
        "User Question",
        "Success Status",
        "Search Queries Attempted",
        "Final Agent Response",
        "Total Run Latency (ms)",
        "Stages Executed Summary",
        "Detailed Node Telemetry (JSON)"
    ]
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    queries_joined = " || ".join([f"'{q}'" for q in query_history]) if query_history else f"'{question}'"
    
    # Compile stage summary
    summary_parts = []
    total_latency = 0
    for node in node_telemetry:
        total_latency += node["latency_ms"]
        token_str = f" [{node['total_tokens']} tks]" if node['total_tokens'] > 0 else ""
        summary_parts.append(f"{node['node']}{token_str} ({node['latency_ms']}ms)")
    stages_summary_str = " -> ".join(summary_parts)

    with open(csv_path, mode="a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(headers)
        
        writer.writerow([
            timestamp,
            question,
            "SUCCESS" if is_successful else "FAILED/UNSUPPORTED",
            queries_joined,
            final_response,
            total_latency,
            stages_summary_str,
            json.dumps(node_telemetry)
        ])
        
    print(f"📊 Rich demo telemetry written to: '{csv_path}'")


# Telemetry Logging for Document Ingestion
def log_ingestion_to_csv(document_name: str, total_chunks: int, total_tokens: int, cost: float, latency_ms: int, csv_path: str = "ingestion_telemetry.csv"):
    file_exists = os.path.exists(csv_path)
    headers = ["Timestamp", "Document Name", "Total Chunks Generated", "Parser Tokens Used", "Ingestion Cost ($)", "Processing Latency (ms)"]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(csv_path, mode="a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow([timestamp, document_name, total_chunks, total_tokens, f"{cost:.6f}", latency_ms])
    print(f"📁 Ingestion telemetry recorded for: '{document_name}'")