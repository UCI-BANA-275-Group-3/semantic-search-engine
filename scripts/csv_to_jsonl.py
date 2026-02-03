#!/usr/bin/env python3
"""Convert labeled CSV back to JSONL for evaluation.

Usage:
  python scripts/csv_to_jsonl.py --input tests/label_queries.csv --output tests/labeled_queries.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json


def main():
    parser = argparse.ArgumentParser(description="Convert labeled CSV to JSONL.")
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        with open(args.output, "w", encoding="utf-8") as out:
            for row in reader:
                query = row.get("query", "").strip()
                relevant_str = row.get("relevant_doc_ids", "").strip()
                if not query or not relevant_str:
                    continue
                relevant = [did.strip() for did in relevant_str.split("|") if did.strip()]
                out.write(json.dumps({"query": query, "relevant": relevant}, ensure_ascii=False) + "\n")

    print(f"Wrote labeled queries to {args.output}")


if __name__ == "__main__":
    main()
