#!/usr/bin/env python3
"""Conservative text cleaning for extracted.jsonl.

This step normalizes whitespace and fixes common PDF artifacts while preserving
provenance fields from the extraction stage.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, Iterable, List


BOILERPLATE_PATTERNS = [
    r"This article was downloaded by:",
    r"Publisher:\s*Institute for Operations Research and the Management Sciences \(INFORMS\)",
    r"INFORMS is located in Maryland, USA",
    r"Publication details, including instructions for authors and subscription information:",
    r"http://pubsonline\.informs\.org",
    r"Published by:\s*The University of Chicago Press",
    r"from the SAGE Social Science Collections\. All Rights Reserved\.",
    r"\*\s*http://jqueryui\.com",
    r"\*\s*Includes:\s*sortable\.css, core\.css, datepicker\.css, slider\.css, theme\.css",
    r"@media\s*\(max-width:\s*\d+px\)\s*\{\}",
    r"R\s*E\s*S\s*E\s*A\s*R\s*C\s*H\s*A\s*R\s*T\s*I\s*C\s*L\s*E",
    r"S\s*P\s*E\s*C\s*I\s*A\s*L\s*I\s*S\s*S\s*U\s*E\s*A\s*R\s*T\s*I\s*C\s*L\s*E",
    r"You are reading copyrighted material.*",
    r"Unauthorized posting, copying, or distributing of this work.*",
    r"This article was downloaded by:.*",
    r"Publisher:\s*Institute for Operations Research and the Management Sciences.*",
    r"INFORMS is located in Maryland, USA.*",
]


UNICODE_FIXES = {
    "\u0007": "$\\theta$",  # bell artifact -> theta
    # Lowercase Greek letters
    "\u03B1": "$\\alpha$",
    "\u03B2": "$\\beta$",
    "\u03B3": "$\\gamma$",
    "\u03B4": "$\\delta$",
    "\u03B5": "$\\epsilon$",
    "\u03B6": "$\\zeta$",
    "\u03B7": "$\\eta$",
    "\u03B8": "$\\theta$",
    "\u03B9": "$\\iota$",
    "\u03BA": "$\\kappa$",
    "\u03BB": "$\\lambda$",
    "\u03BC": "$\\mu$",
    "\u03BD": "$\\nu$",
    "\u03BE": "$\\xi$",
    "\u03BF": "$\\omicron$",
    "\u03C0": "$\\pi$",
    "\u03C1": "$\\rho$",
    "\u03C2": "$\\varsigma$",
    "\u03C3": "$\\sigma$",
    "\u03C4": "$\\tau$",
    "\u03C5": "$\\upsilon$",
    "\u03C6": "$\\phi$",
    "\u03C7": "$\\chi$",
    "\u03C8": "$\\psi$",
    "\u03C9": "$\\omega$",
    # Uppercase Greek letters
    "\u0391": "$A$",
    "\u0392": "$B$",
    "\u0393": "$\\Gamma$",
    "\u0394": "$\\Delta$",
    "\u0395": "$E$",
    "\u0396": "$Z$",
    "\u0397": "$H$",
    "\u0398": "$\\Theta$",
    "\u0399": "$I$",
    "\u039A": "$K$",
    "\u039B": "$\\Lambda$",
    "\u039C": "$M$",
    "\u039D": "$N$",
    "\u039E": "$\\Xi$",
    "\u039F": "$O$",
    "\u03A0": "$\\Pi$",
    "\u03A1": "$P$",
    "\u03A3": "$\\Sigma$",
    "\u03A4": "$T$",
    "\u03A5": "$\\Upsilon$",
    "\u03A6": "$\\Phi$",
    "\u03A7": "$X$",
    "\u03A8": "$\\Psi$",
    "\u03A9": "$\\Omega$",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬀ": "ff",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "’": "'",
    "“": "\"",
    "”": "\"",
    "–": "-",
    "—": "-",
    "‐": "-",
    "‑": "-",
    "\u00A0": " ",
    "…": "...",
}


REFERENCE_HEADERS = [
    r"\bReferences\b",
    r"\bBibliography\b",
]

METADATA_PATTERNS = [
    r"^Volume Title:.*$",
    r"^Volume Authors/Editors:.*$",
    r"^Volume Publisher:.*$",
    r"^Volume ISBNs?:.*$",
    r"^Volume URL:.*$",
    r"^Conference Date:.*$",
    r"^Publication Date:.*$",
    r"^Chapter URL:.*$",
    r"^Chapter pages in book:.*$",
    r"^ISBNs?:.*$",
    r"^URL:.*$",
    r"^http[s]?://\\S+.*$",
]


def _apply_unicode_fixes(text: str) -> str:
    for bad, good in UNICODE_FIXES.items():
        text = text.replace(bad, good)
    return text


def _strip_boilerplate(text: str) -> str:
    for pattern in BOILERPLATE_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return text

def _strip_metadata_lines(text: str) -> str:
    lines = text.splitlines()
    kept: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            kept.append(line)
            continue
        if any(re.match(p, stripped, flags=re.IGNORECASE) for p in METADATA_PATTERNS):
            continue
        kept.append(line)
    return "\n".join(kept)


def _split_references(text: str) -> Dict[str, str]:
    pattern = re.compile("|".join(REFERENCE_HEADERS), flags=re.IGNORECASE)
    match = pattern.search(text)
    if not match:
        return {"main": text, "references": ""}
    start = match.start()
    return {"main": text[:start].strip(), "references": text[start:].strip()}


def _normalize_citations(text: str) -> str:
    # Replace common parenthetical citations with a token.
    citation_pattern = re.compile(r"\(([^\\)]*?\\d{4}[^\\)]*?)\)")
    text = citation_pattern.sub(" <CITATION> ", text)
    text = re.sub(r"(?:<CITATION>\\s*){2,}", "<CITATION> ", text)
    return text


def _extract_lines(text: str) -> List[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def _header_footer_lines(pages: Any) -> List[str]:
    if not pages or not isinstance(pages, list):
        return []
    first_lines = []
    last_lines = []
    for page in pages:
        if not isinstance(page, dict):
            continue
        lines = _extract_lines(page.get("text", ""))
        if not lines:
            continue
        # Skip standalone page numbers at boundaries
        if re.match(r"^\\d+$", lines[0]):
            lines = lines[1:]
        if lines and re.match(r"^\\d+$", lines[-1]):
            lines = lines[:-1]
        if not lines:
            continue
        first_lines.append(lines[0])
        last_lines.append(lines[-1])
    return first_lines + last_lines


def _strip_headers_footers(
    text: str,
    pages: Any,
    title: str,
    creators: List[str],
    min_repeats: int = 3,
) -> str:
    if not pages:
        return text
    lines = _header_footer_lines(pages)
    if not lines:
        return text
    counts: Dict[str, int] = {}
    for line in lines:
        counts[line] = counts.get(line, 0) + 1
    candidates = {
        line
        for line, count in counts.items()
        if count >= min_repeats
        or (title and line.lower() in title.lower())
        or any(line.lower() in c.lower() for c in creators)
        or re.match(r"^\\d+\\s+.+", line)
    }
    for line in candidates:
        text = text.replace(line, "")
    return text


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


def clean_text(text: str) -> str:
    """Conservative cleaning: fix hyphenation + normalize whitespace."""
    if not text:
        return ""
    # Fix common hyphenation across line breaks
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Markdown-aware newline normalization:
    # - Preserve blank lines, headings, and list items
    # - Join soft line breaks within paragraphs
    lines = text.splitlines()
    merged: List[str] = []
    buffer: List[str] = []

    def flush_buffer() -> None:
        if buffer:
            merged.append(" ".join(buffer).strip())
            buffer.clear()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            flush_buffer()
            merged.append("")
            continue
        if stripped.startswith("#") or re.match(r"^[-*+]\s+", stripped):
            flush_buffer()
            merged.append(stripped)
            continue
        buffer.append(stripped)

    flush_buffer()
    normalized = "\n".join(merged)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)

    # Light compaction: remove blank line immediately after headings/bold lines
    normalized = re.sub(r"(\n[#].+)\n\n", r"\1\n", normalized)
    normalized = re.sub(r"(\n\*\*[^\n]+\*\*)\n\n", r"\1\n", normalized)
    # Fix hyphenated words split by spaces (e.g., "Decision- Making")
    normalized = re.sub(r"([A-Za-z])-\s+([A-Za-z])", r"\1-\2", normalized)
    # Fix ligature splits like "specifi cally" -> "specifically"
    normalized = re.sub(r"(fi|fl|ff|ffi|ffl)\s+([a-z])", r"\1\2", normalized, flags=re.IGNORECASE)

    return normalized.strip()


def _clean_page_text(text: str) -> str:
    text = _apply_unicode_fixes(text)
    text = _strip_boilerplate(text)
    text = _strip_metadata_lines(text)
    return text


def clean_record(record: Dict[str, Any]) -> Dict[str, Any]:
    raw_text = record.get("raw_text") or ""
    pages = record.get("pages")
    title = record.get("title") or ""
    creators = record.get("creators") or []

    if pages:
        stitched = []
        for page in pages:
            if not isinstance(page, dict):
                continue
            page_text = page.get("text", "")
            page_text = _clean_page_text(page_text)
            stitched.append(page_text)
        text = "\n\n".join(stitched)
        text = re.sub(r"-\n\n", "", text)
        text = re.sub(r"\n\n", " ", text)
    else:
        text = raw_text

    text = _apply_unicode_fixes(text)
    text = _strip_metadata_lines(text)
    text_before_boilerplate = text
    text = _strip_boilerplate(text)
    boilerplate_removed = text != text_before_boilerplate

    text_before_headers = text
    text = _strip_headers_footers(text, pages, title, creators)
    headers_removed = text != text_before_headers

    text = clean_text(text)

    split = _split_references(text)
    main_text = split["main"]
    references_text = split["references"]
    main_before_citations = main_text
    main_text = _normalize_citations(main_text)
    citations_normalized = main_text != main_before_citations

    return {
        **record,
        "cleaned_text": main_text,
        "references_text": references_text,
        "cleaned_len": len(main_text),
        "raw_len": len(raw_text),
        "_metrics": {
            "boilerplate_removed": boilerplate_removed,
            "headers_removed": headers_removed,
            "references_removed": bool(references_text),
            "citations_normalized": citations_normalized,
        },
    }


def main() -> int:
    """CLI entry point for cleaning extracted text."""
    parser = argparse.ArgumentParser(description="Clean extracted text conservatively.")
    parser.add_argument(
        "--in",
        dest="in_path",
        default="corpus/derived/text/extracted.jsonl",
        help="Input extracted JSONL path.",
    )
    parser.add_argument(
        "--out",
        default="corpus/derived/text/cleaned.jsonl",
        help="Output cleaned JSONL path.",
    )
    parser.add_argument(
        "--logs",
        default="corpus/logs",
        help="Directory for cleaning logs.",
    )
    args = parser.parse_args()

    ensure_dir(os.path.dirname(args.out))
    ensure_dir(args.logs)

    records = list(read_jsonl(args.in_path))
    total = len(records)

    try:
        from tqdm import tqdm  # type: ignore

        iterator = tqdm(records, total=total, desc="Cleaning", unit="doc")
    except Exception:
        iterator = records

    outputs: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    summary = {
        "total_records": 0,
        "boilerplate_removed": 0,
        "headers_removed": 0,
        "references_removed": 0,
        "citations_normalized": 0,
    }

    for record in iterator:
        try:
            cleaned = clean_record(record)
            metrics = cleaned.pop("_metrics", {})
            summary["total_records"] += 1
            if metrics.get("boilerplate_removed"):
                summary["boilerplate_removed"] += 1
            if metrics.get("headers_removed"):
                summary["headers_removed"] += 1
            if metrics.get("references_removed"):
                summary["references_removed"] += 1
            if metrics.get("citations_normalized"):
                summary["citations_normalized"] += 1
            outputs.append(cleaned)
        except Exception as exc:
            errors.append(
                {
                    "doc_id": record.get("doc_id"),
                    "attachment_path": record.get("attachment_path"),
                    "error": str(exc),
                }
            )

    write_jsonl(args.out, outputs)
    errors_path = os.path.join(args.logs, "clean_text_errors.jsonl")
    write_jsonl(errors_path, errors)
    summary_path = os.path.join(args.logs, "clean_text_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    example_path = "corpus/derived/text/cleaned_example.json"
    if outputs:
        with open(example_path, "w", encoding="utf-8") as f:
            json.dump(outputs[0], f, ensure_ascii=False, indent=2)

    print(f"Cleaned records: {len(outputs)}")
    print(f"Errors: {len(errors)}")
    print(f"Output: {args.out}")
    print(f"Error log: {errors_path}")
    print(f"Summary: {summary_path}")
    print(f"Example: {example_path}")
    print(
        "Summary counts:",
        summary["boilerplate_removed"],
        "boilerplate_removed |",
        summary["headers_removed"],
        "headers_removed |",
        summary["references_removed"],
        "references_removed |",
        summary["citations_normalized"],
        "citations_normalized",
    )

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
