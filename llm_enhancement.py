import os
from typing import List, Dict
from transformers import pipeline

# Global variable to cache the model so it doesn't reload every time you search
_summarizer_pipeline = None

def get_summarizer():
    """Lazy-load the summarizer to save memory until it's actually called."""
    global _summarizer_pipeline
    if _summarizer_pipeline is None:
        # Using distilbart: it's small (~300MB) and very fast for local CPUs
        _summarizer_pipeline = pipeline(
            "summarization", 
            model="sshleifer/distilbart-cnn-12-6",
            device=-1  # Forces CPU usage; change to 0 if you have a GPU
        )
    return _summarizer_pipeline

def _build_context(results: List[Dict], max_chars_per_chunk: int = 500) -> str:
    """Combines retrieved chunks into a single string for the model."""
    lines = []
    for r in results:
        # Check all common keys to prevent the "empty bullet" issue
        text = (r.get("text") or r.get("content") or r.get("preview") or "").strip()
        doc_id = r.get("docid") or r.get("id") or "Source"
        
        if text:
            # Truncate long chunks so we don't exceed model token limits
            if len(text) > max_chars_per_chunk:
                text = text[:max_chars_per_chunk] + "..."
            lines.append(f"[{doc_id}]: {text}")
    
    return "\n\n".join(lines)

def summarize_top_k(query: str, results: List[Dict]) -> str:
    """
    Main function for Option A: Summarization.
    Uses a local Transformer instead of OpenAI.
    """
    if not results:
        return "No results found to summarize."

    context_str = _build_context(results)
    
    try:
        model = get_summarizer()
        
        # We prompt the model by giving it the query and the data
        input_text = f"Summarize these search results for the query: {query}\n\nResults:\n{context_str}"
        
        # Generate the summary
        summary_output = model(
            input_text, 
            max_length=150, 
            min_length=40, 
            truncation=True
        )
        
        summary_text = summary_output[0]['summary_text']
        
        return (
            "### Semantic Search Summary\n"
            f"{summary_text}\n\n"
            "---\n"
            "*Generated locally using DistilBART.*"
        )

    except Exception as e:
        # Robust fallback so the UI never looks broken
        return (
            f"Note: Local summarizer is initializing or encountered an error. "
            f"Showing top results for: **{query}**\n\n{context_str}"
        )
