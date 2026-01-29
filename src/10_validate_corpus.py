#!/usr/bin/env python3
# # Validate corpus integrity and log missing/duplicate attachments.

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List


SUPPORTED_EXTS = {".pdf", ".html", ".htm", ".txt"}


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


def main() -> int:
    """CLI entry point for validating manifest.jsonl and attachments.

    Validation checks
    -----------------
    1) JSONL integrity: each line parses as JSON.
    2) Required fields: doc_id present for every record.
    3) File existence: attachment_path exists on disk.
    4) Duplicate IDs: doc_id values must be unique.
    5) Minimum corpus size: valid_records >= --min-docs (default 100).
    6) Optional metadata warnings: missing title, creators, or year.
    7) File extension warnings: attachment_ext not in SUPPORTED_EXTS.
    8) Zero-byte warnings: size_bytes == 0.

    Outputs
    -------
    - validation_summary.json: aggregate counts and metrics.
    - validation_errors.jsonl: one line per hard failure (missing file, duplicate ID, etc.).
    - validation_warnings.jsonl: one line per soft warning (missing metadata, unsupported ext).
    """
    parser = argparse.ArgumentParser(description="Validate manifest.jsonl and corpus files.")
    parser.add_argument(
        "--manifest",
        default="corpus/derived/manifest/manifest.jsonl",
        help="Path to manifest.jsonl",
    )
    parser.add_argument("--logs", default="corpus/logs", help="Directory for logs.")
    parser.add_argument("--min-docs", type=int, default=100, help="Minimum required docs.")
    args = parser.parse_args()

    ensure_dir(args.logs)

    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    doc_ids: List[str] = []
    ext_counts: Counter[str] = Counter()
    missing_title = 0
    missing_creators = 0
    missing_year = 0
    missing_files = 0

    total_records = 0
    valid_records = 0

    for record in read_jsonl(args.manifest):
        total_records += 1
        doc_id = record.get("doc_id")
        if not doc_id:
            errors.append({"type": "missing_doc_id", "record": record})
            continue
        doc_ids.append(doc_id)

        path = record.get("attachment_path")
        if not path or not os.path.exists(path):
            missing_files += 1
            errors.append({"type": "missing_file", "doc_id": doc_id, "path": path})
            continue

        ext = (record.get("attachment_ext") or "").lower()
        if ext:
            ext_counts[ext] += 1
        if ext and ext not in SUPPORTED_EXTS:
            warnings.append({"type": "unsupported_ext", "doc_id": doc_id, "ext": ext})

        if not record.get("title"):
            missing_title += 1
            warnings.append({"type": "missing_title", "doc_id": doc_id})
        if not record.get("creators"):
            missing_creators += 1
            warnings.append({"type": "missing_creators", "doc_id": doc_id})
        if not record.get("year"):
            missing_year += 1
            warnings.append({"type": "missing_year", "doc_id": doc_id})

        size_bytes = record.get("size_bytes")
        if size_bytes == 0:
            warnings.append({"type": "zero_byte_file", "doc_id": doc_id, "path": path})

        valid_records += 1

    duplicates = [doc_id for doc_id, count in Counter(doc_ids).items() if count > 1]
    for doc_id in duplicates:
        errors.append({"type": "duplicate_doc_id", "doc_id": doc_id})

    if valid_records < args.min_docs:
        errors.append(
            {"type": "min_docs_not_met", "valid_records": valid_records, "min_docs": args.min_docs}
        )

    summary = {
        "total_records": total_records,
        "valid_records": valid_records,
        "missing_files": missing_files,
        "duplicate_doc_ids": len(duplicates),
        "extension_counts": dict(ext_counts),
        "missing_title_count": missing_title,
        "missing_creators_count": missing_creators,
        "missing_year_count": missing_year,
        "error_count": len(errors),
        "warning_count": len(warnings),
    }

    summary_path = os.path.join(args.logs, "validation_summary.json")
    errors_path = os.path.join(args.logs, "validation_errors.jsonl")
    warnings_path = os.path.join(args.logs, "validation_warnings.jsonl")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    write_jsonl(errors_path, errors)
    write_jsonl(warnings_path, warnings)

    print(f"Summary: {summary_path}")
    print(f"Errors: {errors_path}")
    print(f"Warnings: {warnings_path}")

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
