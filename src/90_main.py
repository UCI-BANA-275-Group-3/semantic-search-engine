#!/usr/bin/env python3
"""
Semantic search CLI with optional LLM enhancement.

Usage:
  python -m src.90_main --query "your query" --topk-file results.jsonl [--enhance summarize] [--k 10]

This script reads pre-computed top-K search results and applies optional LLM enhancements.
"""

import argparse
import json
import logging
import os
import sys
from typing import List, Dict

from src.llm_enhancement import summarize_top_k
from src.similarity_search import search_top_k

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_top_k_from_file(path: str, k: int) -> List[Dict]:
    """Read top-K results from a JSONL file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Results file not found: {path}")
    
    results: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
                if len(results) >= k:
                    break
            except json.JSONDecodeError as e:
                logger.warning(f"Skipped invalid JSON at line {line_no}: {e}")
    
    if not results:
        raise ValueError(f"No valid results found in {path}")
    
    return results


def format_result(result: Dict) -> str:
    """Format a single result for display."""
    rank = result.get("rank", "?")
    score = result.get("score", "?")
    title = result.get("title") or "(no title)"
    year = result.get("year") or ""
    doc_id = result.get("doc_id") or "?"
    chunk_id = result.get("chunk_id") or "?"
    
    # Extract text snippet
    text = (result.get("chunk_text") or result.get("text") or "").strip()
    snippet = text[:300] + ("..." if len(text) > 300 else "")
    
    year_str = f" ({year})" if year else ""
    return (
        f"{rank:2d}. [{score:.4f}] {title}{year_str}\n"
        f"     doc_id={doc_id}, chunk_id={chunk_id}\n"
        f"     {snippet}\n"
    )


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Semantic search CLI with optional LLM enhancement.",
        epilog="Example: python -m src.90_main --query 'machine learning' --topk-file results.jsonl",
    )
    parser.add_argument("--query", required=True, help="User query string.")
    parser.add_argument("--k", type=int, default=5, help="Number of results to display.")
    parser.add_argument(
        "--enhance",
        choices=["none", "summarize"],
        default="none",
        help="Choose LLM enhancement mode (none=raw results only).",
    )
    parser.add_argument(
        "--topk-file",
        type=str,
        default="topk.jsonl",
        help="Path to JSONL file containing top-K results (default: topk.jsonl).",
    )
    
    args = parser.parse_args()

    try:
        # Load pre-computed results
        logger.info(f"Loading results from {args.topk_file}...")
        top_k = load_top_k_from_file(args.topk_file, k=args.k)
        logger.info(f"Loaded {len(top_k)} results")

        # Display raw results
        print("\n" + "=" * 80)
        print("TOP-K SEARCH RESULTS (RAW)")
        print("=" * 80)
        for result in top_k[:args.k]:
            print(format_result(result))

        # Apply LLM enhancement if requested
        if args.enhance == "summarize":
            logger.info("Generating LLM summary...")
            print("\n" + "=" * 80)
            print("SUMMARY OF TOP-K RESULTS (LLM-ENHANCED)")
            print("=" * 80)
            summary = summarize_top_k(args.query, top_k[:args.k])
            print(f"\nQuery: {args.query}\n")
            print(summary)
            print()
    print("\n=== TOP-K RESULTS (RAW) ===\n")
    for r in top_k:
        # Added r.get('doc') to match your screenshot
        doc_label = r.get('docid') or r.get('doc') or r.get('pdffile') or 'N/A'
        print(f"{r.get('rank', '?')}. score={r.get('score', '?')}, doc={doc_label}")
        
        # Added r.get('content') just in case
        text = (r.get("text") or r.get("preview") or r.get("content") or "").strip()
        print(text[:300] + ("..." if len(text) > 300 else ""))
        print("-" * 80)

        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
