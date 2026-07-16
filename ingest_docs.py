import os
import glob
import time
from dotenv import load_dotenv
from langchain_community.callbacks import get_openai_callback

from utils.telemetry import log_ingestion_to_csv
from ingestion.parsers import extract_structural_fingerprint, chunk_document_hybrid
from ingestion.agent import determine_document_strategy
from ingestion.vector_store import initialize_and_upsert

# primary script for ingesting PDF documents stored in the /data folder
def main():
    load_dotenv()
    
    print("\n" + "="*80)
    print("📁 BATCH INGESTION SYSTEM PIPELINE")
    print("Preparing Layout-Aware Chunking Strategy & Vector Upserts...")
    print("="*80)

    # Locate all PDFs in /data folder
    pdf_paths = glob.glob(os.path.join("data", "*.pdf"))
    if not pdf_paths:
        print("❌ No PDF files found in the '/data' folder. Please place your 10-Q and 10-K files there.")
        return

    print(f"Found {len(pdf_paths)} documents for ingestion: {[os.path.basename(p) for p in pdf_paths]}")

    for pdf_path in pdf_paths:
        file_name = os.path.basename(pdf_path)
        print("\n" + "-"*50)
        print(f"🚀 Processing Document: '{file_name}'")
        print("-"*50)
        
        start_time = time.time()

        # Structural Extraction (Fingerprint)
        print("🔍 Step 1: Generating structural PDF fingerprint...")
        fingerprint = extract_structural_fingerprint(pdf_path)
        
        # Query AI Strategy Agent & Track Tokens/Cost
        print("🤖 Step 2: Querying LLM Strategy Agent...")
        with get_openai_callback() as cb:
            strategy = determine_document_strategy(fingerprint)
            
        print(f"   ✓ Strategy Decision: [{strategy.chunking_strategy}]")
        print(f"   ✓ Type: {strategy.document_type} | Size: {strategy.chunk_size} | Overlap: {strategy.chunk_overlap}")
        print(f"   💬 Reasoning: {strategy.reasoning}")
        print(f"   💵 Strategy Cost: ${cb.total_cost:.6f} | Tokens: {cb.total_tokens}")

        # Parse and Chunk Document based on Agent Strategy
        print("✂️ Step 3: Parsing & slicing document layout...")
        chunks = chunk_document_hybrid(
            pdf_path=pdf_path, 
            chunk_size=strategy.chunk_size, 
            chunk_overlap=strategy.chunk_overlap
        )
        print(f"   ✓ Generated {len(chunks)} structural context chunks.")

        # Vector Generation & Pinecone Upload
        print("⚡ Step 4: Initializing Pinecone index and batch uploading vectors...")
        initialize_and_upsert(chunks, strategy)

        # Track Ingestion Duration
        latency_ms = int((time.time() - start_time) * 1000)
        print(f"⏱️ Finished processing '{file_name}' in {latency_ms / 1000:.2f} seconds.")

        # Log metrics to ingestion_telemetry.csv
        log_ingestion_to_csv(
            document_name=file_name,
            total_chunks=len(chunks),
            total_tokens=cb.total_tokens,
            cost=cb.total_cost,
            latency_ms=latency_ms
        )

    print("\n" + "="*80)
    print("🏁 ALL FILES INGESTED SUCCESSFULLY")
    print("Ingestion telemetries saved directly to 'ingestion_telemetry.csv'.")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()