# TODO: Implement cosine similarity search over embeddings.
#!/usr/bin/env python3
"""
Cosine similarity search over embedded corpus.

Inputs:
- corpus/derived/embeddings/vectors.npy
- corpus/derived/embeddings/meta.jsonl

Query:
- embeds query using the same SentenceTransformers model
- computes cosine similarity (dot product if vectors are normalized)

Outputs:
- prints top-K results to stdout (and optional JSON)
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from .embedding_backends import get_backend


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSONL at line {line_no} in {path}: {exc}") from exc


def load_index(vectors_path: str, meta_path: str) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    if not os.path.exists(vectors_path):
        raise SystemExit(f"Missing vectors file: {vectors_path}")
    if not os.path.exists(meta_path):
        raise SystemExit(f"Missing meta file: {meta_path}")

    vectors = np.load(vectors_path).astype(np.float32, copy=False)
    meta = list(read_jsonl(meta_path))

    if vectors.ndim != 2:
        raise SystemExit(f"vectors.npy must be 2D, got {vectors.shape}")
    if vectors.shape[0] != len(meta):
        raise SystemExit(f"Alignment mismatch: vectors rows={vectors.shape[0]} meta rows={len(meta)}")

    return vectors, meta


def topk_similarities(query_vec: np.ndarray, vectors: np.ndarray, k: int) -> np.ndarray:
    """
    Return indices of top-k most similar vectors.
    Assumes vectors are normalized; cosine similarity = dot product.
    """
    if query_vec.ndim != 1:
        query_vec = query_vec.reshape(-1)
    if vectors.shape[1] != query_vec.shape[0]:
        raise SystemExit(f"Dim mismatch: vectors dim={vectors.shape[1]} query dim={query_vec.shape[0]}")

    scores = vectors @ query_vec  # (N,)
    k = max(1, min(k, scores.shape[0]))
    # argpartition is faster than full sort
    idx = np.argpartition(-scores, k - 1)[:k]
    # sort top-k
    idx = idx[np.argsort(-scores[idx])]
    return idx


def format_result(rank: int, score: float, row: Dict[str, Any]) -> str:
    title = row.get("title") or "(no title)"
    year = row.get("year") or ""
    chunk_id = row.get("chunk_id")
    doc_id = row.get("doc_id")
    creators = row.get("creators") or []
    author = creators[0] if isinstance(creators, list) and creators else ""
    snippet = (row.get("chunk_text") or "")[:300].replace("\n", " ").strip()
    return (
        f"{rank:02d}. score={score:.4f} | {title} {f'({year})' if year else ''}\n"
        f"    doc_id={doc_id} chunk_id={chunk_id} author={author}\n"
        f"    snippet: {snippet}..."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Similarity search over embedded chunks.")
    parser.add_argument("--query", required=True, help="Search query text.")
    parser.add_argument("--k", type=int, default=10, help="Top-K chunks to return.")
    parser.add_argument(
        "--dedup-docs",
        type=int,
        default=5,
        help="Also show top-N unique documents (dedup by doc_id). Set 0 to disable.",
    )
    parser.add_argument(
        "--vectors",
        default="corpus/derived/embeddings/vectors.npy",
        help="Path to vectors.npy",
    )
    parser.add_argument(
        "--meta",
        default="corpus/derived/embeddings/meta.jsonl",
        help="Path to meta.jsonl",
    )
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformers model name (must match embedding model).",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to save results as JSON.",
    )
    args = parser.parse_args()

    vectors, meta_rows = load_index(args.vectors, args.meta)

    backend = get_backend(model_name=args.model, normalize=True, device=None)
    qvec = backend.embed_query(args.query).astype(np.float32, copy=False)

    idx = topk_similarities(qvec, vectors, args.k)
    scores = (vectors[idx] @ qvec).astype(float)

    # Attach chunk text for printing; meta.jsonl doesn't include it by default in our embed step.
    # If you want snippet printing, add chunk_text to meta.jsonl in embed_corpus.py.
    # For now, we print metadata only. (We still support snippet if you later include it.)
    results: List[Dict[str, Any]] = []
    for rank, (i, s) in enumerate(zip(idx.tolist(), scores.tolist()), start=1):
        row = dict(meta_rows[i])
        row["score"] = s
        row["rank"] = rank
        results.append(row)

    # Print top-K chunks
    print("\n=== Top-K CHUNKS ===")
    for r in results:
        title = r.get("title") or "(no title)"
        year = r.get("year") or ""
        print(f"{r['rank']:02d}. score={r['score']:.4f} | {title} {f'({year})' if year else ''}")
        print(f"    doc_id={r.get('doc_id')} chunk_id={r.get('chunk_id')}")

    # Optional: top unique docs
    if args.dedup_docs and args.dedup_docs > 0:
        seen = set()
        unique_docs: List[Dict[str, Any]] = []
        for r in results:
            did = r.get("doc_id")
            if did in seen:
                continue
            seen.add(did)
            unique_docs.append(r)
            if len(unique_docs) >= args.dedup_docs:
                break

        print("\n=== Top UNIQUE DOCS (dedup by doc_id) ===")
        for j, r in enumerate(unique_docs, start=1):
            title = r.get("title") or "(no title)"
            year = r.get("year") or ""
            print(f"{j:02d}. score={r['score']:.4f} | {title} {f'({year})' if year else ''}")
            print(f"    doc_id={r.get('doc_id')}")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(
                {"query": args.query, "k": args.k, "results": results},
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"\nSaved JSON results: {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
