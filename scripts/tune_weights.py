#!/usr/bin/env python3
"""Quick weight tuner for search_top_k.

Runs a small grid search over alpha/lex/meta weights and reports Precision@10.
"""
from __future__ import annotations

import json
from itertools import product
from statistics import mean
from typing import Dict, List, Set

from src.similarity_search import search_top_k


def read_queries(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def precision_at_k(pred: List[str], rel: Set[str], k: int) -> float:
    return sum(1 for p in pred[:k] if p in rel) / k if k else 0.0


def eval_weights(queries_path: str, k: int, alpha: float, lex: float, meta: float) -> float:
    precisions = []
    for q in read_queries(queries_path):
        query_text = q.get("query")
        relevant = set(q.get("relevant", []))
        results = search_top_k(query_text, k, dedup_docs=True, alpha=alpha, lex_weight=lex, meta_weight=meta)
        pred_ids = [r.get("doc_id") or r.get("docid") for r in results]
        p = precision_at_k(pred_ids, relevant, k)
        precisions.append(p)
    return mean(precisions) if precisions else 0.0


if __name__ == "__main__":
    queries = "tests/sample_queries.jsonl"
    k = 10
    alphas = [0.9, 0.75, 0.6, 0.5]
    lexs = [0.0, 0.2, 0.4, 0.6]
    metas = [0.0, 0.05, 0.1]

    best = (0.0, None)
    print("Tuning weights for Precision@10...")
    for a, l, m in product(alphas, lexs, metas):
        p = eval_weights(queries, k, a, l, m)
        print(f"alpha={a:.2f} lex={l:.2f} meta={m:.2f} -> P@10={p*100:.2f}%")
        if p > best[0]:
            best = (p, (a, l, m))

    print("\nBEST:")
    print(f"Precision@10={best[0]*100:.2f}% with weights alpha,lex,meta={best[1]}")