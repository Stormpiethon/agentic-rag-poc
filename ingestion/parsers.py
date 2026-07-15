"""
PDF parsing and structural fingerprint extraction module.
This module provides functionality to open a PDF document, extract its structural fingerprint, 
and return a concise representation of the document's layout, metadata, and sample text. 
The extracted fingerprint is used to inform the agent's decision-making process regarding the 
optimal chunking strategy for the document.
"""

import re
import fitz
from typing import List, Dict, Any


# Get the structural data about the doc to provide context to agent for chunking strategy.
def extract_structural_fingerprint(pdf_path: str) -> str:
    """
    Opens a PDF and extracts a highly compressed 'fingerprint' 
    optimized to handle text/table variance without burning token costs.
    """
    fingerprint_text = []
    
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        
        fingerprint_text.append(f"--- DOCUMENT METADATA ---")
        fingerprint_text.append(f"Source File: {pdf_path}")
        fingerprint_text.append(f"Total Page Count: {total_pages}\n")
        
        # Cover Page Sample (Page 1)
        fingerprint_text.append("--- SAMPLE: COVER PAGE ---")
        fingerprint_text.append(doc[0].get_text("text")[:800])
        
        # Table of Contents Sample (Pages 2-4)
        fingerprint_text.append("\n--- SAMPLE: EARLY PAGES / TOC ---")
        toc_pages = range(1, min(4, total_pages))
        for page_num in toc_pages:
            fingerprint_text.append(doc[page_num].get_text("text")[:1000])
            
        # Multi-Point Body Sampling (Prevents getting trapped in a single giant table)
        # We sample a page at the 25% mark and the 50% mark of the document.
        fingerprint_text.append("\n--- SAMPLE: DEEPER BODY LAYOUTS ---")
        quarter_page = total_pages // 4
        mid_page = total_pages // 2
        
        for sample_page in [quarter_page, mid_page]:
            if sample_page not in toc_pages and sample_page < total_pages:
                fingerprint_text.append(f"[Page {sample_page + 1} Snippet]")
                fingerprint_text.append(doc[sample_page].get_text("text")[:800])
            
        full_fingerprint = "\n".join(fingerprint_text)
        return full_fingerprint[:4000] # Safe token roof (~1000 tokens)
        
    except Exception as e:
        return f"Extraction Error: Unable to parse document structure. Details: {str(e)}"
    

# Parses the PDF document by structural boundaries and semantic chunks within those boundaries
def chunk_document_hybrid(pdf_path: str, chunk_size: int, chunk_overlap: int) -> List[Dict[str, Any]]:
    doc = fitz.open(pdf_path)
    chunks = []
    
    # Reconstruct the document as a list of page texts
    # We keep track of the active section header (e.g., 'Item 1A') as we read
    current_section = "Cover/Intro"
    
    # SEC Item pattern matches (e.g., "Item 1.", "Item 1A.", "PART I - Item 2")
    item_pattern = re.compile(r'^\s*(PART\s+[I|II|III]+)?\s*(Item\s+\d+[A-Z]?)\.?\s*', re.IGNORECASE)

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        
        # Split page into lines to scan for structural section headers
        lines = text.split('\n')
        page_buffer = []
        
        for line in lines:
            stripped = line.strip()
            # If the line looks like an SEC Item heading, update our active metadata tracker
            match = item_pattern.match(stripped)
            # Avoid matching random paragraphs
            if match and len(stripped) < 100:
                current_section = match.group(0).strip()
            page_buffer.append(line)
            
        full_page_text = "\n".join(page_buffer)
        
        # Slice the page text into our Target Chunk Size with Overlap
        # We perform simple character sliding windows within this structural boundary
        start = 0
        while start < len(full_page_text):
            end = start + chunk_size
            chunk_text = full_page_text[start:end]
            
            # Record the chunk contract
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source": pdf_path,
                    "page": page_num + 1,
                    "section": current_section,
                    "chunk_length": len(chunk_text)
                }
            })
            
            # Slide the window forward, subtracting the overlap bounds
            start += (chunk_size - chunk_overlap)
            
    return chunks