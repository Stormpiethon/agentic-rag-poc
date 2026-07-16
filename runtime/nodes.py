"""
Connection point between the state of the graph and the retrieval tools. This module
provides the implementation for retrieving relevant chunks, grading their relevance, 
and rewriting search queries while avoiding duplicates and infinite loops.
"""

import os
from typing import Dict, Any
from openai import OpenAI
from runtime.state import GraphState
from runtime.tools import retrieve_chunks, grade_chunk_relevance, rewrite_search_query


# Retrieves document chunks from Pinecone using the current query string
def retrieve_node(state: GraphState) -> Dict[str, Any]:
    print("\n--- [Node: Retrieve Chunks] ---")
    # If we are looping and have rewritten the query, search with that. Otherwise, use original.
    query_to_use = state.get("current_query") or state["question"]
    
    # Update query history with whatever query we are executing right now
    updated_history = list(state.get("query_history", []))
    if query_to_use not in updated_history:
        updated_history.append(query_to_use)

    retrieved_docs = retrieve_chunks(query_text=query_to_use, top_k=5)
    
    return {
        "documents": retrieved_docs,
        "current_query": query_to_use,
        "query_history": updated_history
    }


# Grades the relevance of retrieved document chunks using LLM-as-a-Judge
def grade_documents_node(state: GraphState) -> Dict[str, Any]:
    print("\n--- [Node: Grade Retrieved Context] ---")
    question = state["question"]
    retrieved_docs = state.get("documents", [])
    
    validated_docs = []
    
    for doc in retrieved_docs:
        # Run the LLM-as-a-Judge to determine if the chunk is relevant to the user's question
        is_relevant = grade_chunk_relevance(query_text=question, document_text=doc["text"])
        if is_relevant:
            validated_docs.append(doc)
            
    print(f"\nGrader Complete: {len(validated_docs)}/{len(retrieved_docs)} chunks approved.")
    
    return {
        "documents": validated_docs
    }


# Rewrites the search query if no relevant documents were found
def rewrite_query_node(state: GraphState) -> Dict[str, Any]:
    print("\n--- [Node: Rewrite Search Query] ---")
    original_question = state["question"]
    history = state.get("query_history", [])
    current_loop_step = state.get("loop_step", 0)
    
    # Pass the query history to the LLM to avoid duplicates and target distinct semantic keywords
    new_query = rewrite_search_query(original_query=original_question, query_history=history)
    
    return {
        "current_query": new_query,
        "loop_step": current_loop_step + 1
    }


# Synthesizes the final answer using the verified, graded documents
def generate_node(state: GraphState) -> Dict[str, Any]:
    print("\n--- [Node: Generate Answer] ---")
    question = state["question"]
    docs = state.get("documents", [])
    
    # Safeguard: If absolutely no documents survived grading and we broke out of the loop
    if not docs:
        final_answer = (
            "I searched the provided financial documents multiple times using alternative keywords, "
            "but I could not find any relevant information to reliably answer your question. "
            "To prevent hallucination, I am unable to provide a response."
        )
        return {"final_response": final_answer}
    
    # Construct context block
    context_blocks = []
    for idx, doc in enumerate(docs):
        source_info = f"Source: {os.path.basename(doc['metadata']['source'])} (Page {doc['metadata']['page']})"
        context_blocks.append(f"[{idx+1}] {source_info}\nContent:\n{doc['text']}")
        
    context_str = "\n\n".join(context_blocks)
    
    # Prompt standard synthesis model
    client = OpenAI()
    system_prompt = (
        "You are an expert financial analyst. Your task is to provide a concise, high-fidelity answer "
        "to the user's question based strictly on the verified financial context provided.\n\n"
        "Rules:\n"
        "1. Ground every single assertion in the retrieved text chunks.\n"
        "2. Do not assume or extrapolate beyond the provided data.\n"
        "3. Cite your sources cleanly using the numbers provided (e.g., [1], [2])."
    )
    
    user_prompt = f"Context:\n{context_str}\n\nQuestion: {question}"
    
    response = client.chat.completions.create(
        # Higher capacity model for synthesis
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        # Drop the temp to keep it very deterministic and avoid hallucinations
        temperature=0.0
    )
    
    final_answer = response.choices[0].message.content.strip()
    return {
        "final_response": final_answer
    }