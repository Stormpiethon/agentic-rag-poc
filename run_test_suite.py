import time
from dotenv import load_dotenv
from utils.telemetry import init_telemetry, log_run_to_csv
from runtime.graph import compiled_graph

# 10 test prompts designed to hit distinct documents and failure paths
TEST_PROMPTS = [
    "What was Apple's research and development expense for the three months ended December 27, 2025?",
    "What was the total cash, cash equivalents, and restricted cash of Apple at the end of the December 27, 2025 period?",
    "What was Apple's total net sales for the fiscal year ended September 27, 2025?",
    "How many shares of common stock did Apple have outstanding as of October 17, 2025?",
    "What was Microsoft's Productivity and Business Processes revenue for the three months ended December 31, 2025?",
    "What was the cash flow from operations for Microsoft during the six months ended December 31, 2025?",
    "What was Microsoft's total cost of revenue for the fiscal year ended June 30, 2025?",
    "How much did Microsoft spend on stock-based compensation expense for the fiscal year ended June 30, 2025?",
    "What was Google's net income for the period ending December 31, 2025?",
    "Compare Apple's net sales directly with the current market capitalization of NVIDIA.",
    "What was Apple's total share-based compensation expense and its related income tax benefit for the three months ended December 27, 2025?",
    "What was Apple's cost of sales for Products versus Services for the three months ended December 27, 2025?",
    "As of December 27, 2025, what was Apple's total unrecognized compensation cost related to outstanding restricted stock units (RSUs)?",
    "What was Apple's total annual spend on Research and Development (R&D) for the fiscal year ended September 27, 2025?",
    "What was Microsoft's quarterly Cloud revenue and its year-over-year constant currency growth rate for the quarter ended June 30, 2025?",
    "What was the total purchase price allocated to the acquisition of Activision Blizzard, and how much goodwill was assigned to Microsoft's More Personal Computing segment from it?",
    "How many shares of common stock did Microsoft have outstanding as of July 24, 2025?",
    "What was Microsoft's Productivity and Business Processes segment revenue for the quarter ended June 30, 2025?",
    "What was Tesla's total capital expenditure for Gigafactory expansions in the fiscal year ended December 31, 2025?",
    "Provide the total research and development expenses for Microsoft for the three months ended December 31, 1999."
]

# Run the test suite of financial questions against the agentic RAG system and log telemetry to CSV
def run_test_suite():
    load_dotenv()
    init_telemetry()
    
    print("\n" + "="*80)
    print("🚦 STARTING BATCH TELEMETRY TEST RUNNER")
    print(f"Queueing {len(TEST_PROMPTS)} financial evaluations across Apple & Microsoft filings...")
    print("="*80 + "\n")
    
    for idx, question in enumerate(TEST_PROMPTS):
        print(f"\n[{idx+1}/{len(TEST_PROMPTS)}] Processing: '{question}'")
        
        initial_state = {
            "question": question,
            "current_query": "",
            "query_history": [],
            "documents": [],
            "loop_step": 0,
            "final_response": "",
            "node_telemetry": []
        }
        
        try:
            # Run the agentic graph
            final_state = compiled_graph.invoke(initial_state)
            
            # Extract success flag (default to True if not present)
            success = final_state.get("is_successful", True)
            
            # Log results to CSV
            log_run_to_csv(
                question = question,
                final_response = final_state["final_response"],
                query_history = final_state.get("query_history", []),
                node_telemetry = final_state["node_telemetry"],
                is_successful = success
            )
            
            # 1-second pause to prevent hitting any rate limits during fast batch executions
            time.sleep(1)
            
        except Exception as e:
            print(f"❌ Error executing query: {e}")

    print("\n" + "="*80)
    print("🏁 TEST RUN COMPLETE")
    print("All telemetry written to 'agent_telemetry.csv'. You are ready for data analysis!")
    print("="*80 + "\n")

if __name__ == "__main__":
    run_test_suite()