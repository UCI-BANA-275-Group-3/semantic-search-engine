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

def _maybe_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Ensure rows are unit-length. Safe even if already normalized."""
    norms = np.linalg.norm(x, axis=1, keepdims=True) + eps
    return x / norms


def aggregate_chunks_to_docs(
    results: List[Dict[str, Any]], agg_method: str = "max"
) -> List[Dict[str, Any]]:
    """
    Aggregate chunk-level scores to document level and return top-K unique documents.
    
    Args:
        results: list of chunk results with 'doc_id' and 'score' fields
        agg_method: 'max' (max chunk score) or 'mean' (mean chunk score per doc)
    
    Returns:
        list of results grouped by doc_id, ranked by aggregated score, with original chunks preserved
    """
    doc_scores: Dict[str, float] = {}
    doc_results: Dict[str, List[Dict[str, Any]]] = {}
    
    for r in results:
        did = r.get("doc_id")
        if not did:
            continue
        score = r.get("score", 0.0)
        
        if did not in doc_results:
            doc_results[did] = []
            if agg_method == "max":
                doc_scores[did] = score
            elif agg_method == "mean":
                doc_scores[did] = score
        else:
            if agg_method == "max":
                doc_scores[did] = max(doc_scores[did], score)
            elif agg_method == "mean":
                doc_scores[did] = (doc_scores[did] * len(doc_results[did]) + score) / (len(doc_results[did]) + 1)
        
        doc_results[did].append(r)
    
    # Rank documents by aggregated score
    ranked_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Build result list with top chunk per document
    aggregated: List[Dict[str, Any]] = []
    for rank, (did, agg_score) in enumerate(ranked_docs, start=1):
        top_chunk = max(doc_results[did], key=lambda x: x.get("score", 0.0))
        row = dict(top_chunk)
        row["score"] = agg_score
        row["rank"] = rank
        row["num_chunks"] = len(doc_results[did])
        aggregated.append(row)
    
    return aggregated


def search_top_k(
    query: str,
    k: int,
    *,
    vectors_path: str = "corpus/derived/embeddings/vectors.npy",
    meta_path: str = "corpus/derived/embeddings/meta.jsonl",
    model: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str | None = None,
    dedup_docs: bool = False,
    agg_method: str = "max",
    alpha: float = 0.6,
    lex_weight: float = 0.3,
    meta_weight: float = 0.1,
) -> List[Dict[str, Any]]:
    """
    Programmatic API: return top-K results as list of dicts with rank + score.
    
    Args:
        query: search query text
        k: number of results to return (chunks if dedup_docs=False, documents if True)
        vectors_path, meta_path, model, device: embedding config
        dedup_docs: if True, aggregate chunk scores to document level and rank by doc
        agg_method: 'max' or 'mean' for aggregation
    
    Returns:
        list of result dicts, each with doc_id, score, rank, etc.
    """
    vectors, meta_rows = load_index(vectors_path, meta_path)

    # Safety: normalize vectors in case someone embedded without normalization
    vectors = _maybe_normalize_rows(vectors.astype(np.float32, copy=False))

    backend = get_backend(model_name=model, normalize=True, device=device)
    qvec = backend.embed_query(query).astype(np.float32, copy=False)

    # Fetch more chunks if deduping, since multiple chunks per doc will be collapsed
    fetch_k = k * 5 if dedup_docs else k
    idx = topk_similarities(qvec, vectors, fetch_k)
    scores = (vectors[idx] @ qvec).astype(float)

    # Pseudo-Relevance Feedback (PRF): expand the query with high-frequency terms
    # from the top semantic candidates to improve recall/precision.
    try:
        # gather top chunk texts
        top_texts = []
        for i in idx.tolist()[: min(5, len(idx))]:
            row_meta = meta_rows[i]
            top_texts.append(str(row_meta.get("chunk_text") or row_meta.get("text") or ""))

        # simple tokenization and stopword filtering
        stopwords = {
            "the", "and", "is", "in", "to", "of", "a", "for", "with", "on", "that", "as", "are",
            "be", "by", "an", "or", "from", "this", "we", "it", "which", "these", "have", "has",
        }
        term_counts: Dict[str, int] = {}
        for txt in top_texts:
            for tok in [t.strip(".,;:()[]\"'`") .lower() for t in txt.split()]:
                if not tok or len(tok) < 3 or tok in stopwords:
                    continue
                term_counts[tok] = term_counts.get(tok, 0) + 1

        # pick top 4 expansion terms not already in query
        q_tokens = set(query.lower().split())
        expansion = []
        for term, _ in sorted(term_counts.items(), key=lambda x: x[1], reverse=True):
            if term in q_tokens:
                continue
            expansion.append(term)
            if len(expansion) >= 4:
                break

        if expansion:
            expanded_query = query + " " + " ".join(expansion)
            qvec2 = backend.embed_query(expanded_query).astype(np.float32, copy=False)
            idx = topk_similarities(qvec2, vectors, fetch_k)
            scores = (vectors[idx] @ qvec2).astype(float)
    except Exception:
        # On any failure, continue with original query results
        pass

    # Compute BM25 idf on the whole corpus (simple, cached in-memory per call)
    try:
        import math

        N = len(meta_rows)
        df: Dict[str, int] = {}
        doc_lens: List[int] = []
        for row in meta_rows:
            text = str(row.get("chunk_text") or row.get("text") or "")
            toks = set([t.strip(".,;:()[]\"'`).lower() for t in text.split() if t])
            for t in toks:
                if len(t) < 3:
                    continue
                df[t] = df.get(t, 0) + 1
            doc_lens.append(len(text.split()))
        avgdl = sum(doc_lens) / max(1, len(doc_lens))
        def idf(term: str) -> float:
            v = df.get(term, 0)
            return math.log((N - v + 0.5) / (v + 0.5) + 1)

        # BM25 params
        k1 = 1.5
        b = 0.75

        # Compute BM25 lexical scores for candidate results
        lex_scores = []
        q_terms = [t.strip(".,;:()[]\"'`).lower() for t in query.split() if t]
        for i in idx.tolist():
            row = meta_rows[i]
            text = str(row.get("chunk_text") or row.get("text") or "")
            tokens = [t.strip(".,;:()[]\"'`).lower() for t in text.split() if t]
            tf: Dict[str, int] = {}
            for tok in tokens:
                if len(tok) < 3:
                    continue
                tf[tok] = tf.get(tok, 0) + 1
            dl = max(1, len(tokens))
            score_bm25 = 0.0
            for qt in q_terms:
                if len(qt) < 2:
                    continue
                t_idf = idf(qt)
                f = tf.get(qt, 0)
                denom = f + k1 * (1 - b + b * (dl / avgdl))
                score_bm25 += t_idf * ((f * (k1 + 1)) / (denom + 1e-9))
            lex_scores.append(float(score_bm25))
    except Exception:
        # fallback to simple token overlap if BM25 stats fail
        lex_scores = []
        for i in idx.tolist():
            row = meta_rows[i]
            text = str(row.get("chunk_text") or row.get("text") or "")
            a = set(query.lower().split())
            b = set(text.lower().split())
            lex_scores.append(float(len(a & b) / max(1, (len(a) + len(b)) / 2)))

    results: List[Dict[str, Any]] = []
    for rank, (i, s) in enumerate(zip(idx.tolist(), scores.tolist()), start=1):
        row = dict(meta_rows[i])
        row["score"] = s
        row["rank"] = rank
        results.append(row)

    # If deduping, aggregate to document level
    if dedup_docs:
        results = aggregate_chunks_to_docs(results, agg_method=agg_method)
    
    # Lightweight lexical reranking + metadata/title boost: combine semantic score with
    # a lexical (TF-IDF or overlap) score and a small metadata/title match boost.
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        texts = [ (r.get("chunk_text") or r.get("text") or "") for r in results ]
        # Fit TF-IDF on candidate texts and transform query
        vec = TfidfVectorizer(stop_words="english").fit(texts + [query])
        qv = vec.transform([query])
        tv = vec.transform(texts)
        lex_scores = cosine_similarity(tv, qv).reshape(-1).astype(float)
    except Exception:
        # Fallback: prefer precomputed BM25-style lexical scores (if available),
        # otherwise fall back to simple token-overlap.
        existing = locals().get("lex_scores")
        if existing and len(existing) == len(results):
            # Use existing lex_scores (BM25) but ensure ordering matches results
            lex_scores = [float(v) for v in existing]
        else:
            def tok_overlap(a: str, b: str) -> float:
                a_tok = set((a or "").lower().split())
                b_tok = set((b or "").lower().split())
                if not a_tok or not b_tok:
                    return 0.0
                return len(a_tok & b_tok) / max(1, (len(a_tok) + len(b_tok)) / 2)

            lex_scores = []
            for r in results:
                text = (r.get("chunk_text") or r.get("text") or "")
                lex_scores.append(float(tok_overlap(text, query)))

    # Compute a simple metadata/title match score
    def meta_match_score(row: Dict[str, Any], q: str) -> float:
        q_tokens = set(q.lower().split())
        score = 0.0
        fields = [row.get("title") or "", " ".join(row.get("creators") or []), row.get("collection_names") or ""]
        text = " ".join([str(f) for f in fields])
        if not text:
            return 0.0
        tks = set(str(text).lower().split())
        inter = q_tokens & tks
        return float(len(inter)) / max(1.0, len(q_tokens))

    # Combine scores (weighted) using provided weights
    for r, lscore in zip(results, lex_scores):
        sem = float(r.get("score", 0.0))
        mscore = meta_match_score(r, query)
        r["lex_score"] = float(lscore)
        r["meta_score"] = float(mscore)
        r["combined_score"] = float(alpha * sem + lex_weight * float(lscore) + meta_weight * float(mscore))

    # Return results sorted by combined_score (descending)
    results = sorted(results, key=lambda x: x.get("combined_score", x.get("score", 0.0)), reverse=True)
    return results[:k]


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
        
        snippet = (r.get("chunk_text") or "")[:300].replace("\n", " ").strip()
        print(f"    snippet: {snippet}...")

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
