"""
PDF parsing and structural fingerprint extraction module.
This module provides functionality to open a PDF document, extract its structural fingerprint, 
and return a concise representation of the document's layout, metadata, and sample text. 
The extracted fingerprint is used to inform the agent's decision-making process regarding the 
optimal chunking strategy for the document.
"""

import fitz


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