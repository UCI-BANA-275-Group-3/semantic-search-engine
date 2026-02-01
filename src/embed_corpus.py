# TODO: Embed chunks and write embeddings.npy plus index.jsonl.
"""
Embed corpus chunks into a dense vector index using SentenceTransformers.

Input:
  - corpus/derived/text/chunks.jsonl  (from 40_chunk_text.py)

Outputs (default):
  - corpus/derived/embeddings/vectors.npy   (float32, shape: [N, D], L2-normalized)
  - corpus/derived/embeddings/meta.jsonl    (per-row metadata for each vector)
  - corpus/derived/embeddings/summary.json  (run summary + validation info)

Design goals:
- production-ready: clear logging, error handling, deterministic output layout
- scalable: streaming read, batch embedding
- compatible with cosine similarity search: normalized vectors
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .embedding_backends import get_backend


# -----------------------------
# IO utils
# -----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


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


def write_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# -----------------------------
# Chunk loading
# -----------------------------
REQUIRED_FIELDS = ("doc_id", "chunk_id", "chunk_index", "chunk_text")


def _validate_record(rec: Dict[str, Any]) -> Optional[str]:
    for k in REQUIRED_FIELDS:
        if k not in rec:
            return f"missing_field:{k}"
    if not str(rec.get("doc_id") or "").strip():
        return "empty_doc_id"
    if not str(rec.get("chunk_id") or "").strip():
        return "empty_chunk_id"
    text = str(rec.get("chunk_text") or "").strip()
    if not text:
        return "empty_chunk_text"
    return None


def load_chunks(path: str) -> Tuple[List[str], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Load chunk_texts and parallel metadata rows (one per chunk).
    Returns (texts, meta_rows, stats).
    """
    texts: List[str] = []
    meta_rows: List[Dict[str, Any]] = []

    total = 0
    kept = 0
    dropped = 0
    drop_reasons: Dict[str, int] = {}

    for rec in read_jsonl(path):
        total += 1
        reason = _validate_record(rec)
        if reason:
            dropped += 1
            drop_reasons[reason] = drop_reasons.get(reason, 0) + 1
            continue

        text = str(rec["chunk_text"]).strip()
        texts.append(text)

        # Keep metadata aligned 1:1 with vectors row order.
        # Include only fields needed by search/UI; keep extra fields if you like.
        meta_rows.append(
            {
                "doc_id": rec.get("doc_id"),
                "chunk_id": rec.get("chunk_id"),
                "chunk_index": rec.get("chunk_index"),
                "chunk_text": rec.get("chunk_text"),
                "token_count": rec.get("token_count"),
                "title": rec.get("title"),
                "creators": rec.get("creators"),
                "year": rec.get("year"),
                "doi": rec.get("doi"),
                "url": rec.get("url"),
                "collection_names": rec.get("collection_names"),
                "attachment_filename": rec.get("attachment_filename"),
                "attachment_path": rec.get("attachment_path"),
                "chunk_start": rec.get("chunk_start"),
                "chunk_end": rec.get("chunk_end"),
                "chunk_len": rec.get("chunk_len"),
            }
        )

        kept += 1

    stats = {
        "input_records": total,
        "kept_records": kept,
        "dropped_records": dropped,
        "drop_reasons": drop_reasons,
    }
    return texts, meta_rows, stats


# -----------------------------
# Embedding
# -----------------------------
def embed_in_batches(
    backend,
    texts: Sequence[str],
    batch_size: int,
) -> np.ndarray:
    """
    Embed texts in batches and stack into (N, D).
    backend.embed_texts already supports batching, but we still chunk here to:
    - keep memory bounded
    - allow progress tracking + resilience
    """
    n = len(texts)
    if n == 0:
        return np.zeros((0, backend.dim), dtype=np.float32)

    all_vecs: List[np.ndarray] = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = texts[start:end]
        vecs = backend.embed_texts(batch, batch_size=batch_size)

        if vecs.ndim != 2:
            raise RuntimeError(f"Embedding output must be 2D, got shape {vecs.shape}")
        if vecs.shape[0] != len(batch):
            raise RuntimeError(
                f"Embedding row count mismatch: expected {len(batch)} got {vecs.shape[0]}"
            )
        all_vecs.append(vecs.astype(np.float32, copy=False))

    return np.vstack(all_vecs)


def main() -> int:
    parser = argparse.ArgumentParser(description="Embed corpus chunks into vectors.npy + meta.jsonl.")
    parser.add_argument(
        "--in",
        dest="in_path",
        default="corpus/derived/text/chunks.jsonl",
        help="Input chunks JSONL path (from 40_chunk_text.py).",
    )
    parser.add_argument(
        "--out-dir",
        default="corpus/derived/embeddings",
        help="Output directory for embeddings artifacts.",
    )
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformers model name.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for embedding.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional device for SentenceTransformers: 'cpu' or 'cuda'.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable L2 normalization (not recommended for cosine similarity).",
    )
    args = parser.parse_args()

    if not os.path.exists(args.in_path):
        raise SystemExit(f"Input not found: {args.in_path}")

    ensure_dir(args.out_dir)

    vectors_path = os.path.join(args.out_dir, "vectors.npy")
    meta_path = os.path.join(args.out_dir, "meta.jsonl")
    summary_path = os.path.join(args.out_dir, "summary.json")

    # 1) Load chunks
    t0 = time.time()
    texts, meta_rows, load_stats = load_chunks(args.in_path)
    if not texts:
        raise SystemExit("No valid chunks found to embed. Check chunks.jsonl generation.")
    load_seconds = time.time() - t0

    # 2) Create backend
    backend = get_backend(
        model_name=args.model,
        normalize=not args.no_normalize,
        device=args.device,
    )

    # 3) Embed
    t1 = time.time()
    vecs = embed_in_batches(backend, texts, batch_size=max(1, args.batch_size))
    embed_seconds = time.time() - t1

    # 4) Validate alignment
    n = len(texts)
    if vecs.shape[0] != n or len(meta_rows) != n:
        raise RuntimeError(
            f"Alignment mismatch: texts={n}, meta={len(meta_rows)}, vectors={vecs.shape}"
        )
    if vecs.dtype != np.float32:
        vecs = vecs.astype(np.float32)

    # 5) Write artifacts
    np.save(vectors_path, vecs)
    write_jsonl(meta_path, meta_rows)

    summary: Dict[str, Any] = {
        "backend": backend.name,
        "model": args.model,
        "normalize": not args.no_normalize,
        "device": args.device,
        "batch_size": args.batch_size,
        "input": {
            "chunks_path": args.in_path,
            **load_stats,
        },
        "output": {
            "out_dir": args.out_dir,
            "vectors_path": vectors_path,
            "meta_path": meta_path,
            "num_vectors": int(vecs.shape[0]),
            "dim": int(vecs.shape[1]),
            "dtype": str(vecs.dtype),
        },
        "timing_seconds": {
            "load": round(load_seconds, 4),
            "embed": round(embed_seconds, 4),
            "total": round(load_seconds + embed_seconds, 4),
        },
        "sanity_checks": {
            "vectors_rows_equal_meta_rows": (vecs.shape[0] == len(meta_rows)),
            "vectors_rows_equal_texts": (vecs.shape[0] == len(texts)),
        },
    }
    write_json(summary_path, summary)

    print(json.dumps(summary, indent=2))
    print(f"Saved vectors: {vectors_path}")
    print(f"Saved meta:    {meta_path}")
    print(f"Saved summary: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
