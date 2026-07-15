"""
Document ingestion agent that uses pydantic to get a structured overview of the incoming document
format so that a chunking strategy can be applied.
"""
from typing import Literal
from openai import OpenAI
from pydantic import BaseModel, Field


# Here is the system prompt passed to the agent every time it is called.
INGESTION_SYSTEM_PROMPT = """
    You are the Lead Data Architect for a governed financial knowledge RAG pipeline. Your objective 
    is to analyze a compressed 'Structural Fingerprint' of an incoming document and determine the 
    optimal layout-aware chunking profile.

    Evaluate the fingerprint across these primary vectors:
    - Header Density: Are there explicit section titles (e.g., 'Item 1A')?
    - Layout Complexity: Does the text layout indicate multiple columns, tabular grids, or standard 
    linear reading paths?
    - Context Sensitivity: Does the terminology contain highly nuanced financial prose requiring 
    parent-child metadata anchoring?

    Execution Rule (Tiered Priority List):
    - Assign 'HYBRID' if the file has clear headers, items, or multi-column layouts where context 
    must be isolated within structural boundaries.
    - Assign 'SEMANTIC' if the text flows sequentially with minimal formatting or section markers.
    - Assign 'STRUCTURAL' if the data is purely schema, code, or dense tabular layouts.
    - Assign 'FIXED' ONLY as a fallback if the document layout appears chaotic, unformatted, or 
    corrupted.

    You must return your analysis strictly matching the requested structural schema.
"""


# This class defines the document ingestion strategy for the agent.
class DocumentStrategy(BaseModel):
    # Extract the document type
    document_type: str = Field(
        description="The detected document classification (e.g., SEC_10K, SEC_10Q, " \
        "QUARTERLY_PRESENTATION, UNKNOWN)."
    )
    # Chunking strategy starts with the most ideal approach then falls back to the next best option based on the document's structure and formatting.
    chunking_strategy: Literal["HYBRID", "SEMANTIC", "STRUCTURAL", "FIXED"] = Field(
        description=(
            "The execution strategy based on your Tiered Priority List:\n"
            "1. HYBRID: Best for structured multi-column or heavily sectioned narrative text (like 10-Ks) where headers act as metadata boundaries.\n"
            "2. SEMANTIC: Best for continuous narrative prose without stark visual hierarchy or clear headers.\n"
            "3. STRUCTURAL: Used if the text is strictly tabular, forms, or data schemas.\n"
            "4. FIXED: The definitive fallback option if formatting is completely unparseable or heavily corrupted."
        )
    )
    # Chunk size related to structure and formatting, measure in tokens.
    chunk_size: int = Field(
        default=1000,
        description=(
            "Recommended character count per chunk slice. "
            "For HYBRID and SEMANTIC, this MUST be between 800 and 1500 to preserve sentence context. "
            "For STRUCTURAL tables, it can be smaller (400-800)."
        )
    )
    # Overlap strategy to prevent contextual clipping between boundaries, measure in tokens.
    chunk_overlap: int = Field(
        default=150,
        description="Recommended character overlap to prevent context clipping. Typically 10% to " \
        "15% of chunk_size (e.g., 100-200 characters)."
    )
    # Agent reasoning for the chosen strategy, providing a concise architectural justification.
    reasoning: str = Field(
        description="A concise architectural justification explaining your choice based on the " \
        "visual layout, column structures, and header frequency."
    )


# This function takes a document fingerprint and returns a strongly-typed DocumentStrategy object
def determine_document_strategy(file_fingerprint: str) -> DocumentStrategy:
    # Initialize the standard OpenAI client
    client = OpenAI()
    
    # Execute the Chat Completion request
    # We use beta.chat.completions.parse to bind the Pydantic class directly to the response
    completion = client.beta.chat.completions.parse(
        # low-cost model for swift structural parsing
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": INGESTION_SYSTEM_PROMPT},
            {"role": "user", "content": f"Please evaluate this document layout and structure:\n\n{file_fingerprint}"}
        ],
        # Forces structural adherence to our Pydantic schema
        response_format = DocumentStrategy,
    )
    
    # Extract and return the strongly-typed Pydantic object
    return completion.choices[0].message.parsed