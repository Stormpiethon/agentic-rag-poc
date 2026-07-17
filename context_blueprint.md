Project Name: Agentic Financial RAG (SEC Financial Report Analyzer)
Database: Pinecone Serverless (Index: "financial-agent-rag")
Embedding Model: text-embedding-3-small (1536 dim)
Ingestion Strategy: Dynamic LLM-as-a-Judge Layout parsing (Custom character limits + 150-char overlap)
State Management Framework: LangGraph

├── data/
│   ├── Apple_10-Q.pdf             # Source financial document (Q1 2026)
│   ├── Microsoft_10-K.pdf         # Source financial document (FY 2025)
│   └── [Other PDFs...]            # Additional SEC filings (Apple, MSFT, etc.)
│
├── ingestion/
│   ├── agent.py                   # LLM Strategy Agent (Classifies layout & returns Pydantic contract)
│   ├── parsers.py                 # PyMuPDF layout parser & hybrid sliding character window chunker
│   ├── vector_store.py            # Pinecone connection, clean idempotent purge, & batch upserts
│   └── strategy_contract.json     # Local trace caching the chosen strategy parameters
│
├── retrieval/
│   └── retriever.py               # Semantic vector queries & metadata filtering interface
│
├── runtime/
│   ├── state.py                   # Shared TypedDict GraphState tracking variables, telemetry, and paths
│   ├── tools.py                   # Core units: LLM-as-a-Judge (Dynamic Entity Alignment), query rewriter
│   ├── graph.py                   # StateGraph physical compilation, nodes, routing logic, and edges
│   └── nodes.py                   # Connection point between the state of the graph and the retrieval tools.
│
├── utils/
│   └── telemetry.py               # Centralized performance, token counting, and CSV row-flattening loggers
│
├── app.py                         # Single-query CLI/interactive runtime interface
├── run_test_suite.py              # Batch-evaluation runner containing 20 granular testing prompts
├── ingestion_telemetry.csv        # Logging trace for document processing (Tokens, cost, latency)
├── agent_telemetry.csv            # Run performance logging trace (Queries attempted, final outputs, latency)
├── requirements.txt               # Project dependency manifesto
└── .gitignore                     # Rigorous version-control exclusions (OneDrive/venv defensive rules)