# TODO: Implement CLI for querying the semantic search pipeline.
# src/90_main.py

import argparse
import json
from typing import List, Dict

from src.llm_enhancement import summarize_top_k

def load_top_k_from_file(path: str, k: int) -> List[Dict]:
    """Read top-K results from a JSONL file."""
    results: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            results.append(json.loads(line))
            if len(results) >= k:
                break
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Semantic search CLI with optional LLM enhancement."
    )
    parser.add_argument("--query", required=True, help="User query string.")
    parser.add_argument("--k", type=int, default=5, help="Number of results to use.")
    parser.add_argument(
        "--enhance",
        choices=["none", "summarize"],
        default="none",
        help="Choose ONE LLM enhancement mode.",
    )
    parser.add_argument(
        "--topk-file",
        type=str,
        required=True,
        help="Path to a JSONL file containing top-K results.",
    )
    args = parser.parse_args()

    query = args.query
    k = args.k

    top_k = load_top_k_from_file(args.topk_file, k=k)

    print("\n=== TOP-K RESULTS (RAW) ===\n")
    for r in top_k:
        print(
            f"{r.get('rank', '?')}. score={r.get('score', '?')}, "
            f"doc={r.get('docid') or r.get('pdffile') or 'N/A'}"
        )
        text = (r.get("text") or r.get("preview") or "").strip()
        print(text[:300] + ("..." if len(text) > 300 else ""))
        print("-" * 80)

    if args.enhance == "summarize":
        summary = summarize_top_k(query, top_k)


        print("\n=== OPTION A: SUMMARY OF TOP-K RESULTS ===\n")
        print(summary)


if __name__ == "__main__":
    main()
