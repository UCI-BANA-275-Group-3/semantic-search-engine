#!/usr/bin/env python3
"""Extract top-N candidate doc_ids per query using existing embeddings.

Writes `tests/sample_queries_candidates.jsonl` with one JSON per line:
  {"query":..., "candidates": [doc_id,...], "scores": [float,...]}

This is read-only: it does not modify embeddings or other pipeline files.
"""
from __future__ import annotations

import json
import os
from typing import List

from src.similarity_search import load_index, _maybe_normalize_rows, topk_similarities
from src.embedding_backends import get_backend


QUERIES_PATH = "tests/sample_queries.jsonl"
OUT_PATH = "tests/sample_queries_candidates.jsonl"
VECTORS = "corpus/derived/embeddings/vectors.npy"
META = "corpus/derived/embeddings/meta.jsonl"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 50


def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main() -> int:
    if not os.path.exists(QUERIES_PATH):
        print(f"Queries file not found: {QUERIES_PATH}")
        return 1

    vectors, meta = load_index(VECTORS, META)
    vectors = _maybe_normalize_rows(vectors.astype("float32", copy=False))

    backend = get_backend(model_name=MODEL, normalize=True, device=None)

    out = []
    for qobj in read_jsonl(QUERIES_PATH):
        query = qobj.get("query") or qobj.get("q") or ""
        if not query:
            continue
        qvec = backend.embed_query(query).astype("float32", copy=False)
        idx = topk_similarities(qvec, vectors, TOP_K)
        scores = (vectors[idx] @ qvec).astype(float).tolist()
        candidates = [meta[i].get("doc_id") for i in idx.tolist()]
        out.append({"query": query, "candidates": candidates, "scores": scores})

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote candidates for {len(out)} queries to {OUT_PATH}")
    # show preview
    for r in out[:3]:
        print(r["query"]) 
        print(r["candidates"][:5])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
