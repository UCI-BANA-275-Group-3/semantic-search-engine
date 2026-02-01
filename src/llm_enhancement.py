# src/llm_enhancement.py

import os
from typing import List, Dict


def _build_context(results: List[Dict], max_chars_per_chunk: int = 400) -> str:
    """Build a text context from top-K results."""
    lines = []
    for r in results:
        label = (
            str(r.get("doc_id"))
            or str(r.get("docid"))
            or str(r.get("pdffile"))
            or str(r.get("chunk_id"))
            or str(r.get("chunkid"))
            or str(r.get("rank"))
            or "source"
        )

        # IMPORTANT: support multiple possible keys
        text = (
            r.get("chunk_text")
            or r.get("text")
            or r.get("preview")
            or r.get("snippet")
            or ""
        )

        if not text:
            continue

        text = str(text).strip()
        if len(text) > max_chars_per_chunk:
            text = text[:max_chars_per_chunk] + "..."

        lines.append(f"[{label}] {text}")

    return "\n\n".join(lines)



def summarize_top_k(query: str, results: List[Dict]) -> str:
    """
    Option A: Summarize top-K results.

    If OPENAI_API_KEY is set, it will call an LLM.
    If not, it returns a pseudo-summary constructed from the chunks.
    """
    if not results:
        return "No results were retrieved, so there is nothing to summarize."

    context_str = _build_context(results)
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        # No key: deterministic summary
        bullets = []
        for i, r in enumerate(results[:5], start=1):
            text = (r.get("chunk_text") or r.get("text") or r.get("preview") or r.get("snippet") or "").strip()
            if len(text) > 160:
                text = text[:160] + "..."
            bullets.append(f"- Result {i}: {text}")
        bullet_str = "\n".join(bullets)
        return (
            "LLM API key is not configured, so this is an auto-generated summary "
            "constructed from the top-K chunks:\n\n"
            f"Query: {query}\n\n"
            f"{bullet_str}"
        )

    # If later your team gets a key, this block will use a real LLM
    try:
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=api_key)
        model_name = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

        prompt = f"""
You are helping a student understand search results from an academic semantic search engine.

User query:
{query}

Below are the top-K relevant text chunks retrieved for this query.
Each chunk may come from a different paper or page.

TASK:
1. Read the chunks.
2. Write a concise summary (4–6 sentences) capturing the main ideas that answer the query.
3. If the sources disagree, briefly mention the disagreement.
4. Use simple, clear language.

Top-K retrieved chunks:
{context_str}
"""

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You summarize semantic search results for an academic research assistant.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=400,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        bullets = []
        for i, r in enumerate(results[:5], start=1):
            text = (r.get("text") or r.get("preview") or "").strip()
            if len(text) > 160:
                text = text[:160] + "..."
            bullets.append(f"- Result {i}: {text}")
        bullet_str = "\n".join(bullets)
        return (
            "Error calling the LLM backend; returning an auto-generated summary "
            "constructed from the top-K chunks instead.\n\n"
            f"Query: {query}\n\n"
            f"{bullet_str}\n\n"
            f"(Internal error: {e})"
        )
# TODO: Implement LLM enhancement (summarize/QA/compare) over top-K results.
