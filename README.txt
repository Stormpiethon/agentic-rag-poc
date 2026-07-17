# Agentic Financial RAG (SEC Financial Report Analyzer)

An enterprise-grade, layout-aware financial document analyzer. The pipeline uses LangGraph 
to govern a multi-step agentic search process featuring dynamic layout-aware chunking, an 
LLM-as-a-Judge relevance grader, adaptive query reformulation, a loop-breaking safety valve, 
and complete operational telemetry.

---

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


## 🛠️ Architecture Deep Dive

### 1. The Ingestion Engine & Strategy Contract
Rather than forcing a one-size-fits-all chunking size on varied financial layouts, the 
ingestion layer is layout-adaptive:
* Structural Fingerprint: ingestion/parsers.py extracts a lightweight 4,000-character 
    "fingerprint" sampling the cover page, TOC, and body sections.
* Strategy Agent: A structured gpt-4o-mini agent analyzes this fingerprint and returns a 
    strongly-typed Pydantic contract (DocumentStrategy).
* The Contracts:
  * HYBRID (e.g., SEC 10-K/10-Q): Targets narrative sections bound by structural headers 
    (Chunk: 1000 char, Overlap: 150 char).
  * STRUCTURAL (e.g., dense spreadsheets or schemas): Drops chunk sizes down to isolate 
    single table objects cleanly (Chunk: 400--800 char).
* Idempotency: Before uploading vectors, the engine performs a metadata filter purge 
    (source == filename) in Pinecone to eliminate duplicate slices.

### 2. Guardrailed Agentic Control Plane (LangGraph)
* Node 1: Retrieve Chunks: Pulls top-5 semantic vector matches matching the active query.
* Node 2: Grade Retrieved Context: An LLM Judge checks the relevance of each individual chunk.
  * Dynamic Entity Alignment: It extracts the core corporate entity of the query and compares 
    it against the "source" metadata field of the chunk. If the query asks for "Tesla" but the 
    chunk comes from "Apple_10-Q.pdf", the chunk is automatically graded FALSE.
* Decision Router:
  * If at least 1 chunk is graded TRUE, route to Generate Answer.
  * If 0 chunks are approved, route to Query Reformulator.
* Node 3: Query Reformulator: Generates a fresh, expanded search query to retry search vector 
    boundaries.
* Loop Breaker: If no context is found after 2 full search cycles, the system safely routes to 
    a graceful failure node, preventing runaway query loops and runaway API costs.

### 3. Production-Grade Telemetry Loggers
Both the ingestion phase and query loops log directly to CSV trackers. To prevent table and 
    paragraph-heavy LLM responses from fragmenting your telemetry CSV files, we implemented 
    an inline sanitizer:
* Line Flattening: The logger strips or substitutes hard carriage returns (\n, \r\n) in the 
    final output string, flattening multi-line text into a single, cohesive CSV field.
* Double-Spacing Block: Opening the file using newline="" prevents operating systems 
    (specifically Windows) from inserting duplicate empty rows.
* Pandas Ready: This guarantees both ingestion_telemetry.csv and agent_telemetry.csv can be 
    loaded into Pandas via pd.read_csv() with absolute zero formatting errors.

---

## 🚀 Getting Started (How to Resume)

When you return to this workspace, follow this exact sequence to spin up and test:

### 1. Set Up the Environment
Create your virtual environment outside the synced scope of OneDrive to avoid locking Git 
or disk execution:
* python -m venv .venv
* source .venv/bin/activate (On Windows use: .venv\Scripts\activate)
* pip install -r requirements.txt

Verify your .env contains:
* OPENAI_API_KEY=your-openai-key
* PINCONE_API_KEY=your-pinecone-key

### 2. Ingest Source Documents
If you add any new financial reports to /data, run the central master driver:
* python ingest_docs.py
This automatically extracts fingerprints, asks the agent to choose the strategy, purges prior 
duplicates in Pinecone, and commits the vectors.

### 3. Run the Evaluation Suite
Verify your graph routing and dynamic entity guardrails by executing your complete 20-query suite:
* python run_test_suite.py

### 4. Build Your Visualizations
Your telemetry data will be clean and ready. To begin your Exploratory Data Analysis (EDA) 
on performance, you can instantly load the logs into your Jupyter Notebook:

```python
import pandas as pd

df_agent = pd.read_csv("agent_telemetry.csv")
df_ingest = pd.read_csv("ingestion_telemetry.csv")

# Quick metric check
print(f"Average System Latency: {df_agent['Total Run Latency (ms)'].mean()} ms")
print(f"Total Operational Cost: ${df_agent['Detailed Node Telemetry (JSON)'].apply(lambda x: sum(n['cost'] for n in json.loads(x))).sum():.4f}")