#!/usr/bin/env python3
"""Extract raw text from manifest attachments into extracted.jsonl.

This stage keeps extraction conservative: it preserves provenance fields from the
manifest and records extraction errors for later review. Cleaning and chunking
are handled in later pipeline steps.
"""

from __future__ import annotations

import argparse
import json
import os
from html.parser import HTMLParser
from typing import Any, Dict, Iterable, List, Optional


SUPPORTED_EXTS = {".pdf", ".html", ".htm", ".txt"}


class _HTMLTextExtractor(HTMLParser):
    """Minimal HTML to text extractor using the standard library."""

    def __init__(self) -> None:
        super().__init__()
        self._chunks: List[str] = []

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if text:
            self._chunks.append(text)

    def get_text(self) -> str:
        return " ".join(self._chunks)


def ensure_dir(path: str) -> None:
    """Create a directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    """Yield JSON objects from a JSONL file."""
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
    """Write a sequence of dicts as JSONL."""
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _normalize_whitespace(text: str) -> str:
    """Light normalization: remove repeated whitespace and trim."""
    return " ".join(text.split()).strip()


def extract_txt(path: str) -> str:
    """Read plain text files."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def extract_html(path: str) -> str:
    """Extract text content from HTML using the stdlib parser."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()
    parser = _HTMLTextExtractor()
    parser.feed(html)
    return parser.get_text()


def extract_pdf(path: str) -> Dict[str, Any]:
    """Extract text from a PDF using PyMuPDF (fitz)."""
    try:
        import fitz  # PyMuPDF
    except Exception as exc:  # pragma: no cover - dependency optional
        raise RuntimeError("PyMuPDF is required for PDF extraction.") from exc

    doc = fitz.open(path)
    pages: List[Dict[str, Any]] = []
    for i in range(len(doc)):
        page = doc.load_page(i)
        text = page.get_text("text") or ""
        pages.append({"page_num": i + 1, "text": text})
    full_text = "\n".join(p["text"] for p in pages)
    return {"text": full_text, "pages": pages}


def extract_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Extract text for a single manifest record and return enriched output."""
    path = record.get("attachment_path")
    ext = (record.get("attachment_ext") or "").lower()

    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Missing attachment: {path}")
    if ext not in SUPPORTED_EXTS:
        raise ValueError(f"Unsupported extension: {ext}")

    if ext == ".pdf":
        pdf_data = extract_pdf(path)
        text = pdf_data["text"]
        pages = pdf_data["pages"]
    elif ext in {".html", ".htm"}:
        text = extract_html(path)
        pages = None
    else:
        text = extract_txt(path)
        pages = None

    return {
        **record,
        "raw_text": text,
        "raw_text_norm": _normalize_whitespace(text),
        "pages": pages,
    }


def main() -> int:
    """CLI entry point for text extraction."""
    parser = argparse.ArgumentParser(description="Extract text from manifest attachments.")
    parser.add_argument(
        "--manifest",
        default="corpus/derived/manifest/manifest.jsonl",
        help="Path to manifest.jsonl",
    )
    parser.add_argument(
        "--out",
        default="corpus/derived/text/extracted.jsonl",
        help="Output extracted JSONL path.",
    )
    parser.add_argument(
        "--logs",
        default="corpus/logs",
        help="Directory for extraction logs.",
    )
    args = parser.parse_args()

    ensure_dir(os.path.dirname(args.out))
    ensure_dir(args.logs)

    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    outputs: List[Dict[str, Any]] = []

    records = list(read_jsonl(args.manifest))
    total = len(records)

    try:
        from tqdm import tqdm  # type: ignore

        iterator = tqdm(records, total=total, desc="Extracting", unit="doc")
    except Exception:
        iterator = records

    for record in iterator:
        try:
            outputs.append(extract_record(record))
        except ValueError as exc:
            # Unsupported types are non-fatal; log and continue.
            warnings.append(
                {
                    "doc_id": record.get("doc_id"),
                    "attachment_path": record.get("attachment_path"),
                    "warning": str(exc),
                }
            )
        except Exception as exc:
            errors.append(
                {
                    "doc_id": record.get("doc_id"),
                    "attachment_path": record.get("attachment_path"),
                    "error": str(exc),
                }
            )

    write_jsonl(args.out, outputs)
    errors_path = os.path.join(args.logs, "extract_text_errors.jsonl")
    warnings_path = os.path.join(args.logs, "extract_text_warnings.jsonl")
    write_jsonl(errors_path, errors)
    write_jsonl(warnings_path, warnings)

    print(f"Extracted records: {len(outputs)}")
    print(f"Errors: {len(errors)}")
    print(f"Warnings: {len(warnings)}")
    print(f"Output: {args.out}")
    print(f"Error log: {errors_path}")
    print(f"Warning log: {warnings_path}")

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
