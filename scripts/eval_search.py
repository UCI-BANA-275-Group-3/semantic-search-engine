#!/usr/bin/env python3
"""Evaluate semantic search quality using labeled queries.

Input: JSONL file with one query per line:
  {"query": "text", "relevant": ["docid1", "docid2", ...]}

Usage:
  python scripts/eval_search.py --queries tests/sample_queries.jsonl --k 10

By default the script computes Precision@k, Recall@k, MAP, MRR, nDCG@k
and prints per-query and averaged metrics. Use `--metric precision` to
report a single metric (e.g. for your "Search Quality 55%" requirement).
"""

from __future__ import annotations

import argparse
import json
from math import log2
from typing import Dict, Iterable, List, Set

from src.similarity_search import search_top_k


def read_queries(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def precision_at_k(pred: List[str], rel: Set[str], k: int) -> float:
    pred_k = pred[:k]
    if k == 0:
        return 0.0
    return sum(1 for p in pred_k if p in rel) / k


def recall_at_k(pred: List[str], rel: Set[str], k: int) -> float:
    if not rel:
        return 0.0
    return sum(1 for p in pred[:k] if p in rel) / len(rel)


def average_precision(pred: List[str], rel: Set[str], k: int) -> float:
    num_hits = 0
    score = 0.0
    for i, p in enumerate(pred[:k], start=1):
        if p in rel:
            num_hits += 1
            score += num_hits / i
    if num_hits == 0:
        return 0.0
    return score / min(len(rel), k)


def mrr(pred: List[str], rel: Set[str]) -> float:
    for i, p in enumerate(pred, start=1):
        if p in rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(pred: List[str], rel: Set[str], k: int) -> float:
    def dcg(scores: List[int]) -> float:
        return sum((2 ** s - 1) / log2(i + 2) for i, s in enumerate(scores))

    gains = [1 if p in rel else 0 for p in pred[:k]]
    ideal = sorted(gains, reverse=True)
    idcg = dcg(ideal)
    if idcg == 0:
        return 0.0
    return dcg(gains) / idcg


def run_eval(queries_path: str, k: int, metric: str = "precision", dedup_docs: bool = True, vectors_path: str = "corpus/derived/embeddings/vectors.npy", meta_path: str = "corpus/derived/embeddings/meta.jsonl", model: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
    queries = list(read_queries(queries_path))
    if not queries:
        print("No queries found in", queries_path)
        return

    precisions = []
    recalls = []
    apks = []
    mrrs = []
    ndcgs = []

    for q in queries:
        query_text = q.get("query")
        relevant = set(q.get("relevant", []))
        # Use programmatic search API with document-level deduplication
        results = search_top_k(query_text, k, dedup_docs=dedup_docs, vectors_path=vectors_path, meta_path=meta_path, model=model)
        pred_ids = [r.get("doc_id") or r.get("docid") or r.get("attachment_filename") for r in results]

        p = precision_at_k(pred_ids, relevant, k)
        rscore = recall_at_k(pred_ids, relevant, k)
        apk = average_precision(pred_ids, relevant, k)
        mmr = mrr(pred_ids, relevant)
        ndcg = ndcg_at_k(pred_ids, relevant, k)

        precisions.append(p)
        recalls.append(rscore)
        apks.append(apk)
        mrrs.append(mmr)
        ndcgs.append(ndcg)

        print(f"Query: {query_text}")
        print(f"  Precision@{k}: {p:.3f}, Recall@{k}: {rscore:.3f}, AP@{k}: {apk:.3f}, MRR: {mmr:.3f}, nDCG@{k}: {ndcg:.3f}")

    from statistics import mean

    p_mean = mean(precisions)
    r_mean = mean(recalls)
    map_k = mean(apks)
    mrr_mean = mean(mrrs)
    ndcg_mean = mean(ndcgs)

    print("\n=== SUMMARY ===")
    print(f"Queries evaluated: {len(queries)}")
    print(f"Dedup mode: {'Document-level' if dedup_docs else 'Chunk-level'}")
    print(f"Model: {model}")
    print(f"Precision@{k} (mean): {p_mean:.3f}")
    print(f"Recall@{k} (mean): {r_mean:.3f}")
    print(f"MAP@{k}: {map_k:.3f}")
    print(f"MRR: {mrr_mean:.3f}")
    print(f"nDCG@{k}: {ndcg_mean:.3f}")

    # Single metric output for project requirement
    if metric == "precision":
        print(f"\nSearch Quality (Precision@{k}): {p_mean * 100:.2f}%")
    elif metric == "map":
        print(f"\nSearch Quality (MAP@{k}): {map_k * 100:.2f}%")
    elif metric == "ndcg":
        print(f"\nSearch Quality (nDCG@{k}): {ndcg_mean * 100:.2f}%")
    else:
        print("\nNo single metric requested. See summary above.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate semantic search against labeled queries.")
    parser.add_argument("--queries", required=True, help="Path to JSONL with queries + relevant doc IDs.")
    parser.add_argument("--k", type=int, default=10, help="Top-K to evaluate.")
    parser.add_argument("--metric", choices=["precision", "map", "ndcg", "none"], default="precision", help="Primary metric to report as 'Search Quality'.")
    parser.add_argument("--no-dedup", action="store_true", help="Disable document-level deduplication (use chunks).")
    parser.add_argument("--vectors", default="corpus/derived/embeddings/vectors.npy", help="Path to vectors.npy")
    parser.add_argument("--meta", default="corpus/derived/embeddings/meta.jsonl", help="Path to meta.jsonl")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="SentenceTransformers model name")
    args = parser.parse_args()

    run_eval(args.queries, args.k, metric=args.metric, dedup_docs=not args.no_dedup, vectors_path=args.vectors, meta_path=args.meta, model=args.model)


if __name__ == "__main__":
    main()
