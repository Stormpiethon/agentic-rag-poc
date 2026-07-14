"""
Every time an agent makes a decision or queries pinecone, a detailed, step-by-step log of 
the decision-making process is generated. This log includes information about the agent's 
reasoning, the data it accessed, and the actions it took.
"""

import os

# Check if the LangSmith environment variables are set for automatic tracing
def init_telemetry():
    if os.getenv("LANGCHAIN_TRACING_V2") == "true" and os.getenv("LANGCHAIN_API_KEY"):
        print("\n !!! LangSmith real-time observability connected automatically via environment hooks.")
    else:
        print("\n !!! Warning: LangSmith API keys missing. Running without real-time tracing.")