#!/usr/bin/env python3
"""Create pseudo-labeled evaluation set using content similarity heuristics.

For each query, we identify potential relevant docs by:
1. Checking if query terms appear in doc text (keyword overlap)
2. High embedding similarity (top candidates)
3. Titles/metadata similarity

This creates a JSONL with reasonable ground-truth estimates for evaluation.
"""

from __future__ import annotations

import argparse
import json
import re
from typing import Any, Dict, Iterable, Set

import numpy as np

from src.similarity_search import load_index, _maybe_normalize_rows
from src.embedding_backends import get_backend


def read_candidates(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def keyword_overlap(query: str, text: str, threshold: float = 0.3) -> bool:
    """Check if query keywords appear significantly in text."""
    query_tokens = set(re.findall(r'\w+', query.lower()))
    text_tokens = set(re.findall(r'\w+', text.lower()))
    
    # Remove common stopwords
    stopwords = {"a", "the", "and", "or", "is", "in", "to", "of", "for", "on", "with", "by"}
    query_tokens -= stopwords
    
    if not query_tokens:
        return False
    
    overlap = len(query_tokens & text_tokens) / len(query_tokens)
    return overlap >= threshold


def create_pseudo_labels(
    candidates_path: str,
    meta_path: str,
    output_path: str,
    threshold_score: float = 0.35,  # top-K candidates with score > this
) -> None:
    """Create pseudo-labeled queries from candidates + heuristics."""
    
    # Load metadata
    meta_rows = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                meta_rows.append(json.loads(line))
    
    doc_meta = {r.get("doc_id"): r for r in meta_rows}
    
    labeled_queries = []
    
    for cand_obj in read_candidates(candidates_path):
        query = cand_obj.get("query", "")
        candidates = cand_obj.get("candidates", [])
        scores = cand_obj.get("scores", [])
        
        if not query or not candidates:
            continue
        
        # Use top candidates as pseudo-relevant
        relevant = set()
        
        for cand_id, score in zip(candidates[:20], scores[:20]):
            if score < threshold_score:
                break
            
            meta = doc_meta.get(cand_id)
            if not meta:
                continue
            
            # Heuristic 1: high embedding score + keyword overlap in title/creators
            title = (meta.get("title") or "").lower()
            creators = " ".join(meta.get("creators") or []).lower()
            combined_text = f"{title} {creators}"
            
            if score > 0.45:  # very high confidence from embeddings
                relevant.add(cand_id)
            elif score > threshold_score and keyword_overlap(query, combined_text, threshold=0.2):
                relevant.add(cand_id)
        
        if len(relevant) >= 1:  # only include if at least 1 relevant
            labeled_queries.append({
                "query": query,
                "relevant": list(relevant),
            })
    
    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        for obj in labeled_queries:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    
    print(f"Created {len(labeled_queries)} pseudo-labeled queries")
    print(f"Average relevant docs per query: {np.mean([len(q['relevant']) for q in labeled_queries]):.1f}")
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Create pseudo-labeled eval set from candidates.")
    parser.add_argument(
        "--candidates",
        default="tests/sample_queries_candidates.jsonl",
        help="Input candidates JSONL",
    )
    parser.add_argument(
        "--meta",
        default="corpus/derived/embeddings/meta.jsonl",
        help="Path to meta.jsonl",
    )
    parser.add_argument(
        "--output",
        default="tests/pseudo_labeled_queries.jsonl",
        help="Output JSONL",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.35,
        help="Score threshold for considering candidates as relevant",
    )
    args = parser.parse_args()
    
    create_pseudo_labels(args.candidates, args.meta, args.output, threshold_score=args.threshold)


if __name__ == "__main__":
    main()
