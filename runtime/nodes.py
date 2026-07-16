"""
Connection point between the state of the graph and the retrieval tools. This module
provides the implementation for retrieving relevant chunks, grading their relevance, 
and rewriting search queries while avoiding duplicates and infinite loops.
"""

import os
import time
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback
from runtime.state import GraphState
from runtime.tools import retrieve_chunks, grade_chunk_relevance, rewrite_search_query


# Helper to track node performance and token usage
def track_node_performance(node_name: str, start_time: float, cb_object=None) -> Dict[str, Any]:
    """Helper to structure node latency and token tracking."""
    latency = int((time.time() - start_time) * 1000)
    tokens = {
        "node": node_name,
        "latency_ms": latency,
        "prompt_tokens": cb_object.prompt_tokens if cb_object else 0,
        "completion_tokens": cb_object.completion_tokens if cb_object else 0,
        "total_tokens": cb_object.total_tokens if cb_object else 0,
        "cost": cb_object.total_cost if cb_object else 0.0
    }
    print(f"   ⏱️ [{node_name}] took {latency}ms | Tokens: {tokens['total_tokens']} (${tokens['cost']:.6f})")
    return tokens

# Retrieves document chunks from Pinecone using the current query string
# Inside retrieve_node in runtime/nodes.py:
def retrieve_node(state: GraphState) -> Dict[str, Any]:
    print("\n--- [Node: Retrieve Chunks] ---")
    start = time.time()
    query_to_use = state.get("current_query") or state["question"]
    
    # Estimate input tokens - tracked locally since langchain only tracks LLM calls, not retrievals
    estimated_tokens = max(1, len(query_to_use) // 4)
    estimated_cost = (estimated_tokens / 1_000_000) * 0.02 # $0.02 per million tokens

    retrieved_docs = retrieve_chunks(query_text=query_to_use, top_k=5)
    
    # Track performance with calculated embedding values
    telemetry_list = list(state.get("node_telemetry", []))
    telemetry_list.append({
        "node": "retrieve_chunks",
        "latency_ms": int((time.time() - start) * 1000),
        "prompt_tokens": estimated_tokens,
        "completion_tokens": 0,
        "total_tokens": estimated_tokens,
        "cost": estimated_cost
    })
    
    return {
        "documents": retrieved_docs,
        "current_query": query_to_use,
        "node_telemetry": telemetry_list
    }


# Grades the relevance of retrieved document chunks using LLM-as-a-Judge
def grade_documents_node(state: GraphState) -> Dict[str, Any]:
    print("\n--- [Node: Grade Retrieved Context] ---")
    start = time.time()
    question = state["question"]
    retrieved_docs = state.get("documents", [])
    
    validated_docs = []
    telemetry_list = list(state.get("node_telemetry", []))
    
    # Wrap LLM evaluations in callback
    with get_openai_callback() as cb:
        for doc in retrieved_docs:
            # Extract the raw file name (e.g., "Apple_10-Q.pdf")
            source_file = os.path.basename(doc["metadata"]["source"])
            
            # Pass the source file name to the grader!
            is_relevant = grade_chunk_relevance(
                query_text=question, 
                document_text=doc["text"], 
                source_name=source_file
            )
            if is_relevant:
                validated_docs.append(doc)
                
    telemetry_list.append(track_node_performance("grade_documents", start, cb))
    print(f"↳ Grader Complete: {len(validated_docs)}/{len(retrieved_docs)} chunks approved.")
    
    return {
        "documents": validated_docs,
        "node_telemetry": telemetry_list
    }


# Rewrites the search query if no relevant documents were found
def rewrite_query_node(state: GraphState) -> Dict[str, Any]:
    print("\n--- [Node: Rewrite Search Query] ---")
    start = time.time()
    original_question = state["question"]
    history = state.get("query_history", [])
    current_loop_step = state.get("loop_step", 0)
    
    telemetry_list = list(state.get("node_telemetry", []))
    
    with get_openai_callback() as cb:
        new_query = rewrite_search_query(original_query=original_question, query_history=history)
        
    telemetry_list.append(track_node_performance("rewrite_query", start, cb))
    
    return {
        "current_query": new_query,
        "loop_step": current_loop_step + 1,
        "node_telemetry": telemetry_list
    }


# Synthesizes the final answer using the verified, graded documents
def generate_node(state: GraphState) -> Dict[str, Any]:
    print("\n--- [Node: Generate Answer] ---")
    start = time.time()
    question = state["question"]
    docs = state.get("documents", [])
    telemetry_list = list(state.get("node_telemetry", []))
    
    if not docs:
        final_answer = (
            "I searched the provided financial documents multiple times using alternative keywords, "
            "but I could not find any relevant information to reliably answer your question."
        )
        telemetry_list.append(track_node_performance("generate_answer_empty", start))
        # Returns False for success
        return {"final_response": final_answer, "node_telemetry": telemetry_list, "is_successful": False}
    
    context_blocks = []
    for idx, doc in enumerate(docs):
        source_info = f"Source: {os.path.basename(doc['metadata']['source'])} (Page {doc['metadata']['page']})"
        context_blocks.append(f"[{idx+1}] {source_info}\nContent:\n{doc['text']}")
        
    context_str = "\n\n".join(context_blocks)
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
    
    system_prompt = (
        "You are an expert financial analyst. Your task is to provide a concise, high-fidelity answer "
        "to the user's question based strictly on the verified financial context provided.\n\n"
        "Rules:\n"
        "1. Ground every assertion in the retrieved text chunks.\n"
        "2. Cite your sources cleanly using the numbers provided."
    )
    user_prompt = f"Context:\n{context_str}\n\nQuestion: {question}"
    
    with get_openai_callback() as cb:
        response = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
    
    final_answer = response.content.strip()
    telemetry_list.append(track_node_performance("generate_answer", start, cb))
    
    return {
        "final_response": final_answer,
        "node_telemetry": telemetry_list,
        "is_successful": True
    }