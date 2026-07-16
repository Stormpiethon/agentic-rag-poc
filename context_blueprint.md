Project Name: Agentic Financial RAG (SEC Financial Report Analyzer)
Database: Pinecone Serverless (Index: "financial-agent-rag")
Embedding Model: text-embedding-3-small (1536 dim)
Ingestion Strategy: Dynamic LLM-as-a-Judge Layout parsing (Custom character limits + 150-char overlap)
State Management Framework: LangGraph

Directory Structure:
├── data/
│   └── Apple_10-Q.pdf            # Source financial document
├── ingestion/
│   ├── parsers.py                # PyMuPDF layout parser & hybrid chunker
│   ├── agent.py                  # LLM Strategy Agent (Classifies layout & returns Pydantic contract)
│   ├── vector_store.py           # Pinecone connection, clean idempotent purge, & batch upserts
│   └── strategy_contract.json    # Cached execution contract for accuracy tracing
├── retrieval/
│   └── retriever.py              # Semantic vector queries & metadata filters
├── runtime/
│   ├── state.py                  # Defines the shared TypedDict GraphState for LangGraph
│   ├── tools.py                  # Isolated functions: retriever, boolean-grade document grader, generalized SEC query rewriter
│   └── graph.py                  # The physical graph compiler (Nodes, edges, entry points)
├── utils/
│   ├── telemetry.py              # Global settings / init
│   └── telemetry_logger.py       # (Future step) Exporting token callback metrics to CSV
└── app.py                        # Entry driver script