#!/usr/bin/env python3
"""Re-embed corpus with a stronger SentenceTransformers model.

Usage:
  python scripts/reembed_corpus.py --model sentence-transformers/multi-qa-mpnet-base-dot-v1 --output-dir corpus/derived/embeddings_v2

This will re-embed all chunks in the corpus with a new model, saving to a new directory.
You can then point search to the new embeddings and compare performance.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Iterable

import numpy as np
from tqdm import tqdm

from src.embedding_backends import get_backend


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def reembed(
    meta_path: str,
    output_dir: str,
    model: str = "sentence-transformers/multi-qa-mpnet-base-dot-v1",
    batch_size: int = 32,
) -> None:
    """Re-embed corpus with stronger model."""
    os.makedirs(output_dir, exist_ok=True)

    backend = get_backend(model_name=model, normalize=True, device=None)
    
    # Read all meta and chunks
    meta_rows = list(read_jsonl(meta_path))
    print(f"Loaded {len(meta_rows)} metadata rows")

    # Embed all chunks at once using the backend
    chunk_texts = [r.get("chunk_text", "") for r in meta_rows]
    embeddings = backend.embed_texts(chunk_texts, batch_size=batch_size)
    
    print(f"Embedded shape: {embeddings.shape}")

    # Save vectors and meta
    vectors_path = os.path.join(output_dir, "vectors.npy")
    meta_out_path = os.path.join(output_dir, "meta.jsonl")

    np.save(vectors_path, embeddings)
    print(f"Saved vectors to {vectors_path}")

    with open(meta_out_path, "w", encoding="utf-8") as f:
        for row in meta_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved meta to {meta_out_path}")

    print(f"\nDone! Use with: python src/similarity_search.py --query 'text' --vectors {vectors_path} --meta {meta_out_path} --model {model}")


def main():
    parser = argparse.ArgumentParser(description="Re-embed corpus with a stronger model.")
    parser.add_argument(
        "--meta",
        default="corpus/derived/embeddings/meta.jsonl",
        help="Path to input meta.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="corpus/derived/embeddings_v2",
        help="Output directory for new embeddings",
    )
    parser.add_argument(
        "--model",
        default="sentence-transformers/multi-qa-mpnet-base-dot-v1",
        help="SentenceTransformers model name (stronger models have better quality but slower)",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for embedding")
    args = parser.parse_args()

    reembed(args.meta, args.output_dir, model=args.model, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
