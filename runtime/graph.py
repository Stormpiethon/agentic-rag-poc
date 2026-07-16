from langgraph.graph import StateGraph, START, END
from runtime.state import GraphState
from runtime.nodes import generate_node, retrieve_node, grade_documents_node, rewrite_query_node


# Conditional routing edges that determine the next node to execute based on the current state
def route_post_grading(state: GraphState) -> str:
    print("\n--- [Routing Decision] ---")
    validated_docs = state.get("documents", [])
    loop_step = state.get("loop_step", 0)
    
    # We have validated context! Proceed straight to answering.
    if len(validated_docs) > 0:
        print("\nDecision: Sufficient context found. Routing to [Generate Answer].")
        return "generate"
        
    # No validated context found, but we've already tried searching twice.
    # Break the loop to prevent infinite runs/costs and force final response generation.
    if loop_step >= 2:
        print(f"\nDecision: No context found after {loop_step} attempts. Max retries hit. Routing to [Generate Answer] (Graceful Failure).")
        return "generate"
        
    # No validated context found, and we still have retry budget left.
    print(f"\nDecision: Context irrelevant. Loop count: {loop_step}/2. Routing to [Rewrite Query].")
    return "rewrite_query"


# =====================================================================
# Initialize the state-aware workflow
workflow = StateGraph(GraphState)

# Register all of our structural nodes
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade_documents", grade_documents_node)
workflow.add_node("rewrite_query", rewrite_query_node)
workflow.add_node("generate", generate_node)

# Set the execution entry point
workflow.set_entry_point("retrieve")

# Draw the standard, deterministic edges
workflow.add_edge("retrieve", "grade_documents")
# Loop rewritten search query back to database
workflow.add_edge("rewrite_query", "retrieve")
workflow.add_edge("generate", END)

# Draw the conditional decision edge exiting the grader
workflow.add_conditional_edges(
    "grade_documents",
    route_post_grading,
    {
        "generate": "generate",
        "rewrite_query": "rewrite_query"
    }
)

# Compile the workflow into a runnable application
compiled_graph = workflow.compile()