"""
Holds the isolated function blocks—specifically the semantic query embedder for 
Pinecone, the LLM-as-a-judge document grader, and the query rewriter prompt.
"""

import os
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from openai import OpenAI
from retrieval.retriever import query_vector_store

# =====================================================================
# LLM-AS-A-JUDGE STRUCTURED SCHEMA
class GradeDocument(BaseModel):
    binary_score: bool = Field(
        description="Is the document chunk relevant to the user query? True if yes, False if no."
    )
    explanation: str = Field(
        description="A brief, single-sentence explanation of why the document chunk is or is not relevant."
    )

# =====================================================================
# ISOLATED TOOL ACTIONS

# Interfaces with Pinecone to retrieve semantically matching vector payloads
def retrieve_chunks(query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
    print(f"[Tool: Retriever] Querying database for: '{query_text}'")
    return query_vector_store(query_text=query_text, top_k=top_k)

# Evaluates if a retrieved text chunk contains information that can help answer the user's question
def grade_chunk_relevance(query_text: str, document_text: str) -> str:
    client = OpenAI()
    
    system_prompt = (
        "You are an elite corporate auditor grading the quality of retrieved financial context.\n"
        "Your task is to determine if the provided text chunk contains any information relevant "
        "to answering the user's query.\n\n"
        "Guidelines:\n"
        "1. Be highly conservative but fair. If the chunk contains even a single sentence, metric, "
        "or detail that could help answer the query, grade it as True.\n"
        "2. If the chunk is completely off-topic, blank, or contains only generic formatting text "
        "that does not apply to the query, grade it as False."
    )
    
    user_prompt = f"User Query: {query_text}\n\nRetrieved Text Chunk:\n{document_text}"
    
    # Force OpenAI to return a strict, Pydantic-conforming JSON object
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format=GradeDocument,
    )
    
    result = completion.choices[0].message.parsed
    print(f"  ↳ Judge Score: [{str(result.binary_score).upper()}] - {result.explanation}")
    return result.binary_score

# Rewrites the user's search query and avoids duplicates to target distinct semantic keywords
def rewrite_search_query(original_query: str, query_history: List[str]) -> str:
    client = OpenAI()
    
    history_str = "\n".join([f"- {q}" for q in query_history])
    
    system_prompt = (
        "You are an expert financial search engineer. Your job is to rewrite search queries "
        "to optimize semantic vector matching against SEC financial reporting documents.\n"
        "You will look at the user's original intent and the history of search terms that have "
        "already been tried and failed.\n\n"
        "Rules:\n"
        "1. Output ONLY the raw rewritten search query. Do not include introductory text, quotes, or markdown.\n"
        "2. Focus on core accounting terminology, numbers, items, or specific regulatory tables."
    )
    
    user_prompt = (
        f"Original User Question: {original_query}\n\n"
        f"Prior Search Queries Tried (Failed to find relevant chunks):\n{history_str}\n\n"
        f"Generate a brand new, unique query that tackles the question from a different financial perspective:"
    )
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7 
    )
    
    rewritten_query = response.choices[0].message.content.strip()
    print(f"[Tool: Query Rewriter] Reformulated search query to: '{rewritten_query}'")
    return rewritten_query