#!/usr/bin/env python3
"""Convert candidates JSONL to CSV for easy manual labeling.

Usage:
  python scripts/create_labeling_csv.py --candidates tests/sample_queries_candidates.jsonl --output tests/label_queries.csv

Then open label_queries.csv in Excel/Google Sheets and mark relevant doc_ids for each query.
Save as CSV, then run: python scripts/csv_to_jsonl.py --input tests/label_queries.csv --output tests/labeled_queries.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
from typing import Any, Dict


def main():
    parser = argparse.ArgumentParser(description="Create CSV for labeling queries.")
    parser.add_argument(
        "--candidates",
        default="tests/sample_queries_candidates.jsonl",
        help="Input candidates JSONL",
    )
    parser.add_argument(
        "--output",
        default="tests/label_queries.csv",
        help="Output CSV file",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Only show top N candidates per query",
    )
    args = parser.parse_args()

    rows = []
    with open(args.candidates, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            query = obj.get("query", "")
            candidates = obj.get("candidates", [])[:args.top_n]
            
            rows.append({
                "query": query,
                "candidate_doc_ids": "|".join(candidates),
                "relevant_doc_ids": "",  # To be filled in by labeler
                "notes": "",
            })

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["query", "candidate_doc_ids", "relevant_doc_ids", "notes"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Created {args.output} with {len(rows)} queries.")
    print("Instructions: Fill in 'relevant_doc_ids' column (pipe-separated doc_ids that are relevant to each query).")


if __name__ == "__main__":
    main()
