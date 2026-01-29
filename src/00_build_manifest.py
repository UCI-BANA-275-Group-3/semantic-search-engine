#!/usr/bin/env python3
# Build a manifest.jsonl from Zotero Better BibLaTeX JSON exports.

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple


SUPPORTED_EXTS = {".pdf", ".html", ".htm", ".txt"}


def load_json(path: str) -> Dict[str, Any]:
    """Load a JSON file from disk.

    Parameters
    ----------
    path : str
        Path to the JSON file.

    Returns
    -------
    Dict[str, Any]
        Parsed JSON object (dict root expected for Better BibLaTeX exports).
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str) -> None:
    """Create a directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def safe_year(date_str: Optional[str]) -> Optional[int]:
    """Extract a 4-digit year from a Zotero date string.

    Examples
    --------
    "2021-05" -> 2021
    "01/2022" -> 2022
    """
    if not date_str:
        return None
    match = re.search(r"(\d{4})", date_str)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def normalize_creators(creators: Optional[List[Dict[str, Any]]]) -> List[str]:
    """Convert Zotero creator dicts into 'Last, First' strings."""
    if not creators:
        return []
    names: List[str] = []
    for c in creators:
        if not isinstance(c, dict):
            continue
        first = c.get("firstName")
        last = c.get("lastName")
        if last and first:
            names.append(f"{last}, {first}")
        elif last:
            names.append(str(last))
        elif first:
            names.append(str(first))
    return names


def normalize_tags(tags: Optional[List[Any]]) -> List[str]:
    """Normalize Zotero tags to a flat list of strings."""
    if not tags:
        return []
    out: List[str] = []
    for t in tags:
        if isinstance(t, dict) and "tag" in t:
            out.append(str(t["tag"]))
        elif isinstance(t, str):
            out.append(t)
    return out


def parse_attachment_path(
    path_value: Optional[str], storage_root: str
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Parse Zotero attachment paths to extract key, filename, and full path.

    Parameters
    ----------
    path_value : Optional[str]
        Zotero path like "files/<ATTACHMENT_KEY>/<filename>".
    storage_root : str
        Root directory where attachments are stored.
    """
    if not path_value:
        return None, None, None
    path_value = path_value.replace("\\", "/")
    match = re.search(r"files/([^/]+)/(.+)", path_value)
    if not match:
        return None, None, None
    attachment_key = match.group(1)
    filename = match.group(2)
    full_path = os.path.join(storage_root, attachment_key, filename)
    return attachment_key, filename, full_path


def file_size(path: Optional[str]) -> Optional[int]:
    """Return the file size in bytes, or None if unavailable."""
    if not path or not os.path.exists(path):
        return None
    try:
        return os.path.getsize(path)
    except OSError:
        return None


def infer_mime(ext: str) -> Optional[str]:
    """Infer a minimal MIME type from a file extension."""
    ext = ext.lower()
    if ext == ".pdf":
        return "application/pdf"
    if ext in {".html", ".htm"}:
        return "text/html"
    if ext == ".txt":
        return "text/plain"
    return None


def build_collection_lookup(collections: Dict[str, Any]) -> Dict[int, List[Dict[str, str]]]:
    """Create a mapping from Zotero itemID to collection metadata."""
    item_to_collections: Dict[int, List[Dict[str, str]]] = {}
    for key, col in collections.items():
        if not isinstance(col, dict):
            continue
        item_ids = col.get("items", [])
        for item_id in item_ids:
            if not isinstance(item_id, int):
                continue
            entry = {"key": key, "name": col.get("name", "")}
            item_to_collections.setdefault(item_id, []).append(entry)
    return item_to_collections


def extract_item_key(value: Optional[str]) -> Optional[str]:
    """Extract a Zotero item key from a URI or select string."""
    if not value:
        return None
    match = re.search(r"/items/([A-Z0-9]+)", value)
    if match:
        return match.group(1)
    return None


def normalize_zotero_path(path_value: Optional[str]) -> Optional[str]:
    """Normalize Zotero attachment paths to a consistent 'files/<key>/<name>' form."""
    if not path_value:
        return None
    return path_value.replace("\\", "/")


def build_attachment_parent_lookup(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Map attachment keys and paths to parent item dicts for metadata bridging.

    Keys used:
    - attachment itemKey from uri/select
    - attachment path ("files/<ATTACHMENT_KEY>/<filename>")
    """
    attachment_to_parent: Dict[str, Dict[str, Any]] = {}
    for item in items:
        attachments = item.get("attachments", [])
        if not isinstance(attachments, list):
            continue
        for att in attachments:
            if not isinstance(att, dict):
                continue
            att_key = (
                extract_item_key(att.get("uri"))
                or extract_item_key(att.get("select"))
                or att.get("itemKey")
            )
            if att_key:
                attachment_to_parent[att_key] = item
            att_path = normalize_zotero_path(att.get("path"))
            if att_path:
                attachment_to_parent[att_path] = item
    return attachment_to_parent


def stable_doc_id(item_key: Optional[str], attachment_key: Optional[str], attachment_path: str) -> str:
    """Generate a stable doc_id for a manifest record."""
    if item_key and attachment_key:
        return f"{item_key}:{attachment_key}"
    if item_key:
        return item_key
    digest = hashlib.sha1(attachment_path.encode("utf-8")).hexdigest()[:16]
    return f"hash:{digest}"


def iter_attachment_records(
    items: List[Dict[str, Any]],
    item_to_collections: Dict[int, List[Dict[str, str]]],
    attachment_to_parent: Dict[str, Dict[str, Any]],
    source_name: str,
    storage_root: str,
) -> Iterable[Dict[str, Any]]:
    """Yield manifest records for each attachment found in the metadata.

    Handles both:
    - top-level attachment items (itemType == 'attachment')
    - regular items with an embedded 'attachments' list
    """
    for idx, item in enumerate(items):
        item_type = item.get("itemType")
        item_key = item.get("itemKey")
        item_id = item.get("itemID")

        base_item = item
        if item_type == "attachment":
            parent_key = extract_item_key(item.get("uri")) or extract_item_key(item.get("select"))
            parent_path = normalize_zotero_path(item.get("path"))
            parent_item = (
                attachment_to_parent.get(parent_key or "")
                or attachment_to_parent.get(parent_path or "")
            )
            if parent_item:
                base_item = parent_item
                item_key = parent_item.get("itemKey") or item_key
                item_id = parent_item.get("itemID") or item_id

        creators = normalize_creators(base_item.get("creators"))
        tags = normalize_tags(base_item.get("tags"))
        year = safe_year(base_item.get("date"))
        title = base_item.get("title")
        abstract = base_item.get("abstractNote")
        doi = base_item.get("DOI")
        url = base_item.get("url")
        language = base_item.get("language")
        date_added = base_item.get("dateAdded")
        date_modified = base_item.get("dateModified")

        collections = item_to_collections.get(item_id, [])
        collection_keys = [c.get("key", "") for c in collections]
        collection_names = [c.get("name", "") for c in collections]

        # Case 1: Attachment is a top-level item.
        if item_type == "attachment" and item.get("path"):
            attachment_key, filename, full_path = parse_attachment_path(
                item.get("path"), storage_root
            )
            if not full_path or not filename:
                continue
            ext = os.path.splitext(filename)[1].lower()
            mime = infer_mime(ext)
            record = {
                "doc_id": stable_doc_id(item_key, attachment_key, full_path),
                "item_key": item_key,
                "attachment_key": attachment_key,
                "title": title,
                "creators": creators,
                "year": year,
                "collection_keys": collection_keys,
                "collection_names": collection_names,
                "source_metadata": source_name,
                "source_item_index": idx,
                "attachment_path": full_path,
                "attachment_filename": filename,
                "attachment_ext": ext,
                "mime": mime,
                "size_bytes": file_size(full_path),
                "abstract": abstract,
                "tags": tags,
                "doi": doi,
                "url": url,
                "date_added": date_added,
                "date_modified": date_modified,
                "language": language,
            }
            yield record
            continue

        # Case 2: Main item with embedded attachments list.
        attachments = item.get("attachments", [])
        for att in attachments:
            if not isinstance(att, dict):
                continue
            attachment_key, filename, full_path = parse_attachment_path(
                att.get("path"), storage_root
            )
            if not full_path or not filename:
                continue
            ext = os.path.splitext(filename)[1].lower()
            mime = infer_mime(ext)
            record = {
                "doc_id": stable_doc_id(item_key, attachment_key, full_path),
                "item_key": item_key,
                "attachment_key": attachment_key,
                "title": title,
                "creators": creators,
                "year": year,
                "collection_keys": collection_keys,
                "collection_names": collection_names,
                "source_metadata": source_name,
                "source_item_index": idx,
                "attachment_path": full_path,
                "attachment_filename": filename,
                "attachment_ext": ext,
                "mime": mime,
                "size_bytes": file_size(full_path),
                "abstract": abstract,
                "tags": tags,
                "doi": doi,
                "url": url,
                "date_added": date_added,
                "date_modified": date_modified,
                "language": language,
            }
            yield record


def main() -> int:
    """CLI entry point for building the manifest JSONL file."""
    parser = argparse.ArgumentParser(description="Build manifest.jsonl from Zotero metadata.")
    parser.add_argument(
        "--metadata",
        default="corpus/raw/zotero/metadata/library.json",
        help="Path to Better BibLaTeX JSON export.",
    )
    parser.add_argument(
        "--storage-root",
        default="corpus/raw/zotero/storage",
        help="Root directory for Zotero attachment storage.",
    )
    parser.add_argument(
        "--out",
        default="corpus/derived/manifest/manifest.jsonl",
        help="Output manifest JSONL path.",
    )
    parser.add_argument(
        "--logs",
        default="corpus/logs",
        help="Directory for logs.",
    )
    args = parser.parse_args()

    ensure_dir(os.path.dirname(args.out))
    ensure_dir(args.logs)

    data = load_json(args.metadata)
    items = data.get("items", [])
    collections = data.get("collections", {})
    if not isinstance(items, list):
        raise SystemExit("Metadata 'items' must be a list.")
    if not isinstance(collections, dict):
        collections = {}

    item_to_collections = build_collection_lookup(collections)
    attachment_to_parent = build_attachment_parent_lookup(items)
    source_name = os.path.basename(args.metadata)

    missing_attachments_log = os.path.join(args.logs, "manifest_missing_attachments.jsonl")
    total = 0
    written = 0

    with open(args.out, "w", encoding="utf-8") as out_f, open(
        missing_attachments_log, "w", encoding="utf-8"
    ) as miss_f:
        for record in iter_attachment_records(
            items, item_to_collections, attachment_to_parent, source_name, args.storage_root
        ):
            total += 1
            if not os.path.exists(record["attachment_path"]):
                miss_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"Manifest records written: {written} (from {total} attachments)")
    print(f"Missing attachment log: {missing_attachments_log}")
    print(f"Manifest path: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
