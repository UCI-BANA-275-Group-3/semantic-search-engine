# TODO: Chunk cleaned text deterministically into chunks.jsonl.
"""
Chunk cleaned text deterministically into chunks.jsonl

Input:  corpus/derived/text/cleaned.jsonl   (from 30_clean_text.py)
Output: corpus/derived/text/chunks.jsonl    (one JSON object per chunk)

Best-quality choices:
- Uses a real tokenizer (Hugging Face) for accurate token counts and overlap.
- Paragraph-aware first; falls back to sentence splitting for oversized paragraphs.
- Deterministic chunk IDs: {doc_id}::chunk000123
- Adds token overlap (default 80) to improve retrieval quality at boundaries.
- Enforces a final max token cap AFTER overlap to stay under model limits (e.g., 512).
- Avoids tokenizer warnings by NOT tokenizing very large blocks just to measure length.
- Emits one JSONL line per chunk (ideal for embedding + cosine similarity).
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


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
                raise SystemExit(f"Invalid JSONL at line {line_no}: {exc}") from exc


def write_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# -----------------------------
# Tokenization (best quality)
# -----------------------------
@dataclass
class TokenizerWrapper:
    """
    Wraps a HF tokenizer with deterministic encode/decode.
    Falls back to approximate token count if transformers isn't available.
    """

    model_name: str

    def __post_init__(self) -> None:
        self._tok = None
        try:
            from transformers import AutoTokenizer  # type: ignore

            self._tok = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        except Exception:
            self._tok = None

    def available(self) -> bool:
        return self._tok is not None

    def encode(self, text: str) -> List[int]:
        if not self._tok:
            # Approx fallback: words * 1.33
            words = len(re.findall(r"\S+", text or ""))
            return list(range(int(words * 1.33)))
        return self._tok.encode(text or "", add_special_tokens=False)

    def decode(self, token_ids: List[int]) -> str:
        if not self._tok:
            return ""
        return self._tok.decode(token_ids, skip_special_tokens=True).strip()

    def count(self, text: str) -> int:
        return len(self.encode(text))


def _cap_to_max_tokens(tok: TokenizerWrapper, text: str, max_tokens: int) -> str:
    """
    Ensure text is <= max_tokens tokens. Uses tokenizer encode/decode when available.
    If tokenizer isn't available, returns text unchanged.
    """
    text = (text or "").strip()
    if not text:
        return text
    if not tok.available():
        return text
    ids = tok.encode(text)
    if len(ids) <= max_tokens:
        return text
    return tok.decode(ids[:max_tokens]).strip()


# -----------------------------
# Text structure helpers
# -----------------------------
def _normalize_newlines(text: str) -> str:
    return (text or "").replace("\r\n", "\n").replace("\r", "\n")


def _split_paragraphs(text: str) -> List[str]:
    """
    Split into paragraph-like blocks.
    Keeps Markdown headings/lists as separate blocks to preserve structure.
    """
    text = _normalize_newlines(text).strip()
    if not text:
        return []

    # Ensure headings start new paragraph
    text = re.sub(r"\n(#+\s+)", r"\n\n\1", text)

    # Keep list items separated
    text = re.sub(r"\n(\s*[-*+]\s+)", r"\n\n\1", text)

    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return parts


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"\'])")


def _split_sentences(block: str) -> List[str]:
    """
    Conservative sentence splitter: deterministic, no extra deps.
    """
    block = (block or "").strip()
    if not block:
        return []
    # crude abbreviation protection
    block = re.sub(
        r"\b(e\.g|i\.e|et al)\.\s+",
        lambda m: m.group(0).replace(". ", "._ "),
        block,
        flags=re.IGNORECASE,
    )
    sents = _SENT_SPLIT.split(block)
    sents = [s.replace("._ ", ". ").strip() for s in sents if s.strip()]
    return sents


def _is_probably_too_long_by_chars(text: str, token_limit: int) -> bool:
    """
    Cheap guard to avoid tokenizing extremely long strings (which triggers warnings).
    Rough heuristic: ~1 token ≈ 4 characters (English-ish).
    """
    if not text:
        return False
    return len(text) > token_limit * 4


def _make_chunk_id(doc_id: str, chunk_index: int) -> str:
    return f"{doc_id}::chunk{chunk_index:06d}"


# -----------------------------
# Chunking core (token-based)
# -----------------------------
@dataclass(frozen=True)
class ChunkingConfig:
    # Strong defaults for MiniLM-like models (max ~512 tokens)
    target_tokens: int = 320
    overlap_tokens: int = 80
    hard_max_tokens: int = 480
    min_tokens: int = 80


def _best_effort_char_span(
    original: str, chunk_text: str, start_from: int
) -> Tuple[Optional[int], Optional[int], int]:
    """
    Find chunk_text in original starting at start_from. Returns (start, end, next_search_pos).
    If not found, returns (None, None, start_from).
    """
    if not original or not chunk_text:
        return None, None, start_from
    idx = original.find(chunk_text, start_from)
    if idx == -1:
        return None, None, start_from
    return idx, idx + len(chunk_text), idx + len(chunk_text)


def _build_overlap_prefix_by_tokens(tok: TokenizerWrapper, prev_chunk_text: str, overlap_tokens: int) -> str:
    """
    Build an overlap prefix from the END of previous chunk using token IDs, then decode.
    """
    if overlap_tokens <= 0 or not prev_chunk_text.strip():
        return ""
    if not tok.available():
        return ""
    ids = tok.encode(prev_chunk_text)
    if len(ids) <= overlap_tokens:
        return prev_chunk_text.strip()
    tail = ids[-overlap_tokens:]
    return tok.decode(tail).strip()


def chunk_document(cleaned_text: str, tok: TokenizerWrapper, cfg: ChunkingConfig) -> List[str]:
    """
    Return list of chunk texts.
    Token-based + paragraph-aware + overlap + final max-cap.
    """
    text = _normalize_newlines(cleaned_text).strip()
    if not text:
        return []

    blocks = _split_paragraphs(text)
    if not blocks:
        return []

    chunks: List[str] = []
    cur_parts: List[str] = []
    cur_tokens = 0

    def flush() -> None:
        nonlocal cur_parts, cur_tokens
        if not cur_parts:
            return
        chunk = "\n\n".join(cur_parts).strip()
        if chunk:
            chunks.append(chunk)
        cur_parts = []
        cur_tokens = 0

    for block in blocks:
        # Avoid tokenizer warnings by not encoding huge blocks just to measure length.
        if _is_probably_too_long_by_chars(block, 512):
            bt = cfg.hard_max_tokens + 1  # force splitting path
        else:
            bt = tok.count(block)

        # If block itself is huge, split into sentences and pack
        if bt > cfg.hard_max_tokens:
            flush()
            sents = _split_sentences(block)
            if not sents:
                # Fallback: hard split by tokens if tokenizer available
                if tok.available():
                    ids = tok.encode(block)
                    start = 0
                    while start < len(ids):
                        end = min(start + cfg.hard_max_tokens, len(ids))
                        piece = tok.decode(ids[start:end]).strip()
                        if piece:
                            chunks.append(piece)
                        start = end
                else:
                    # Char fallback (rare)
                    step = 2000
                    for i in range(0, len(block), step):
                        piece = block[i : i + step].strip()
                        if piece:
                            chunks.append(piece)
                continue

            buf: List[str] = []
            buf_tokens = 0
            for s in sents:
                st = tok.count(s) if not _is_probably_too_long_by_chars(s, 512) else cfg.hard_max_tokens + 1

                # If sentence is still too big, split by tokens
                if st > cfg.hard_max_tokens and tok.available():
                    ids = tok.encode(s)
                    start = 0
                    while start < len(ids):
                        end = min(start + cfg.hard_max_tokens, len(ids))
                        piece = tok.decode(ids[start:end]).strip()
                        if piece:
                            chunks.append(piece)
                        start = end
                    continue

                if buf_tokens + st > cfg.target_tokens and buf:
                    chunks.append(" ".join(buf).strip())
                    buf = [s]
                    buf_tokens = st
                else:
                    buf.append(s)
                    buf_tokens += st
            if buf:
                chunks.append(" ".join(buf).strip())
            continue

        # Normal block packing
        if cur_tokens + bt > cfg.target_tokens and cur_parts:
            flush()
            cur_parts = [block]
            cur_tokens = bt
        else:
            cur_parts.append(block)
            cur_tokens += bt

    flush()

    # Apply overlap (prefix from previous chunk)
    if cfg.overlap_tokens > 0 and len(chunks) > 1:
        overlapped: List[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            prefix = _build_overlap_prefix_by_tokens(tok, overlapped[-1], cfg.overlap_tokens)
            if prefix:
                overlapped.append((prefix + "\n\n" + chunks[i]).strip())
            else:
                overlapped.append(chunks[i])
        chunks = overlapped

    # Enforce final max token length AFTER overlap (prevents > model max like 512)
    final_max = cfg.hard_max_tokens
    if tok.available() and getattr(tok._tok, "model_max_length", None):
        m = int(tok._tok.model_max_length)
        if m and 0 < m < 1_000_000:
            # keep a small safety buffer
            final_max = min(final_max, max(128, m - 16))

    chunks = [_cap_to_max_tokens(tok, ch, final_max) for ch in chunks]

    # Drop tiny chunks (token-based)
    final: List[str] = []
    for ch in chunks:
        if tok.count(ch) >= cfg.min_tokens:
            final.append(ch.strip())
    return final


def _pack_chunk_record(
    base: Dict[str, Any],
    chunk_text: str,
    chunk_index: int,
    tok: TokenizerWrapper,
    char_start: Optional[int],
    char_end: Optional[int],
) -> Dict[str, Any]:
    doc_id = str(base.get("doc_id") or "unknown")
    ct = chunk_text.strip()
    return {
        "doc_id": doc_id,
        "chunk_id": _make_chunk_id(doc_id, chunk_index),
        "chunk_index": chunk_index,
        "chunk_text": ct,
        "chunk_len": len(ct),
        "token_count": tok.count(ct),
        "chunk_start": char_start,
        "chunk_end": char_end,
        # forward metadata
        "title": base.get("title"),
        "creators": base.get("creators"),
        "year": base.get("year"),
        "doi": base.get("doi"),
        "url": base.get("url"),
        "collection_names": base.get("collection_names"),
        "attachment_filename": base.get("attachment_filename"),
        "attachment_path": base.get("attachment_path"),
    }


# -----------------------------
# CLI
# -----------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Chunk cleaned text into chunks.jsonl (token-based).")
    parser.add_argument(
        "--in",
        dest="in_path",
        default="corpus/derived/text/cleaned.jsonl",
        help="Input cleaned JSONL path (from 30_clean_text.py).",
    )
    parser.add_argument(
        "--out",
        default="corpus/derived/text/chunks.jsonl",
        help="Output chunks JSONL path.",
    )
    parser.add_argument("--logs", default="corpus/logs", help="Directory for logs.")
    parser.add_argument(
        "--tokenizer-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HF tokenizer model name (match embedding model for best chunking).",
    )

    parser.add_argument("--target-tokens", type=int, default=320, help="Target tokens per chunk.")
    parser.add_argument("--overlap-tokens", type=int, default=80, help="Overlap tokens between chunks.")
    parser.add_argument("--hard-max-tokens", type=int, default=480, help="Hard max tokens per chunk.")
    parser.add_argument("--min-tokens", type=int, default=80, help="Drop chunks smaller than this token count.")

    args = parser.parse_args()

    if not os.path.exists(args.in_path):
        raise SystemExit(f"Input not found: {args.in_path}")

    ensure_dir(os.path.dirname(args.out))
    ensure_dir(args.logs)

    tok = TokenizerWrapper(model_name=args.tokenizer_model)
    cfg = ChunkingConfig(
        target_tokens=args.target_tokens,
        overlap_tokens=args.overlap_tokens,
        hard_max_tokens=args.hard_max_tokens,
        min_tokens=args.min_tokens,
    )

    outputs: List[Dict[str, Any]] = []
    total_docs = 0
    total_chunks = 0
    dropped_small = 0
    missing_text = 0

    for rec in read_jsonl(args.in_path):
        total_docs += 1
        doc_id = rec.get("doc_id")
        cleaned = (rec.get("cleaned_text") or "").strip()

        if not doc_id or not cleaned:
            missing_text += 1
            continue

        chunks = chunk_document(cleaned, tok, cfg)

        search_pos = 0
        for i, ch in enumerate(chunks):
            cstart, cend, search_pos = _best_effort_char_span(cleaned, ch, search_pos)
            if tok.count(ch) < cfg.min_tokens:
                dropped_small += 1
                continue
            outputs.append(_pack_chunk_record(rec, ch, i, tok, cstart, cend))
            total_chunks += 1

    write_jsonl(args.out, outputs)

    summary = {
        "input_docs": total_docs,
        "output_chunks": total_chunks,
        "dropped_small_chunks": dropped_small,
        "docs_missing_text_or_id": missing_text,
        "tokenizer_model": args.tokenizer_model,
        "tokenizer_available": tok.available(),
        "params": {
            "target_tokens": cfg.target_tokens,
            "overlap_tokens": cfg.overlap_tokens,
            "hard_max_tokens": cfg.hard_max_tokens,
            "min_tokens": cfg.min_tokens,
        },
        "out_path": args.out,
    }

    summary_path = os.path.join(args.logs, "chunk_text_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, indent=2))

    if not tok.available():
        print("WARNING: transformers tokenizer not available. Install 'transformers' for best-quality token chunking.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
