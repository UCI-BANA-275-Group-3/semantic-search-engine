# Semantic Search Engine (Team Project)

## Overview
Production-ready semantic search over a document corpus (≥100 docs) using embeddings + cosine similarity retrieval, plus one LLM enhancement mode.

**Domain:** Academic papers (PDFs) exported from Zotero
**Interface:** Command line (CLI)
**Bonus (optional):** Streamlit / Gradio / HF Space

---

## Status Note (Existing Corpus)
All Zotero exports should live under `corpus/raw/zotero/` as documented below.

---

## Repository Structure (Source of Truth)

```
semantic-search/
  README.md
  ARCHITECTURE.md
  TEAM_CONTRIBUTIONS.md
  requirements.txt

  corpus/
    raw/
      zotero/
        metadata/
          library.json            # Better BibLaTeX JSON export
        storage/
          <ATTACHMENT_KEY>/
            *.pdf
    derived/
      manifest/
        manifest.jsonl
      text/
        extracted.jsonl
        cleaned.jsonl
      chunks/
        chunks.jsonl
      embeddings/
        embeddings.npy
        index.jsonl

  src/
    00_build_manifest.py
    10_validate_corpus.py
    20_extract_text.py
    30_clean_text.py
    40_chunk_text.py
    50_embedding_backends.py
    60_embed_corpus.py
    70_similarity_search.py
    80_llm_enhancement.py
    90_main.py
```

**Notes**
- Files under `corpus/derived/` are generated artifacts.
- `ARCHITECTURE.md` documents design choices and pipeline details.
- `TEAM_CONTRIBUTIONS.md` lists who did what for peer evaluation.

---

## Pipeline Stages (Mapped to Source)

### **Manifest building**
- Script: `src/00_build_manifest.py`
- Inputs: `corpus/raw/zotero/metadata/library.json`, `corpus/raw/zotero/storage/**`
- Output: `corpus/derived/manifest/manifest.jsonl`

Completed Tasks: Build the manifest (00_build_manifest.py)

Define manifest schema (one record per document/attachment) and required fields (doc_id, title, creators, year, collection, attachment_path, mime, source_json, zotero_key, etc.).
Parse library.json (Better BibLaTeX JSON) to extract items + collections.
Resolve each item’s attachment(s) to corpus/raw/zotero/storage/<ATTACHMENT_KEY>/....
Normalize paths and generate a stable doc_id (e.g., hash of attachment path or item key + attachment key).
Write manifest.jsonl with one line per resolved attachment; log unresolved items to corpus/logs/.

### **Corpus validation**
- Script: `src/10_validate_corpus.py`
- Inputs: manifest + raw storage
- Outputs: `corpus/logs/validation_summary.json`, `corpus/logs/validation_errors.jsonl`, `corpus/logs/validation_warnings.jsonl`


Completed Tasks: Validate the corpus (10_validate_corpus.py)

Read the manifest and perform checks:
Paths exist and are readable
MIME types are supported (PDF/HTML)
No duplicate doc_ids
Minimum doc count ≥ 100
Produce a summary report (counts, missing files, duplicates) and failure logs in corpus/logs/.
Exit non‑zero if critical checks fail (missing attachments, too few docs), to gate the rest of the pipeline.

### **Text extraction**
- Script: `src/20_extract_text.py`
- Input: `corpus/derived/manifest/manifest.jsonl`
- Output: `corpus/derived/text/extracted.jsonl`

Completed Tasks: Text extraction (20_extract_text.py)
1) Read manifest.jsonl and iterate attachments with supported extensions.
2) Extract raw text (PDF/HTML/TXT) while capturing page numbers when possible.
3) Clean minimal artifacts (line breaks/hyphenation), preserve provenance fields.
4) Write extracted.jsonl with one record per attachment (or per page/section if needed).
5) Log extraction errors and warnings to corpus/logs/ (e.g., unsupported types, missing files).

### **Cleaning**
- Script: `src/30_clean_text.py`
- Input: `corpus/derived/text/extracted.jsonl`
- Output: `corpus/derived/text/cleaned.jsonl`

### **Chunking**
- Script: `src/40_chunk_text.py`
- Input: `corpus/derived/text/cleaned.jsonl`
- Output: `corpus/derived/chunks/chunks.jsonl`

### **Embeddings**
- Scripts: `src/50_embedding_backends.py`, `src/60_embed_corpus.py`
- Input: `corpus/derived/chunks/chunks.jsonl`
- Outputs: `corpus/derived/embeddings/embeddings.npy`, `corpus/derived/embeddings/index.jsonl`

### **Similarity search**
- Script: `src/70_similarity_search.py`
- Inputs: embeddings artifacts
- Output: top-K chunks with cosine scores

### **LLM enhancement (choose ONE)**
- Script: `src/80_llm_enhancement.py`
- Modes: summarize / QA / compare

### **CLI interface**
- Script: `src/90_main.py`
- Runs retrieval + optional enhancement

---

## Setup

### Python
- Recommended: Python 3.12

### Install dependencies
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Extraction dependencies (add as we grow)
- **PDFs:** `pymupdf` (required by `src/20_extract_text.py`)
- **HTML/TXT:** stdlib only for now
- **Progress:** `tqdm` (optional status bar in extraction)

### Environment variables (if using OpenAI)
Create `.env` (do not commit):
- `OPENAI_API_KEY=...`

---

## Running (Planned Commands)
```bash
python -m src.00_build_manifest \
  --zotero-metadata corpus/raw/zotero/metadata/library.json \
  --zotero-storage corpus/raw/zotero/storage \
  --out corpus/derived/manifest/manifest.jsonl

python -m src.20_extract_text \
  --manifest corpus/derived/manifest/manifest.jsonl \
  --out corpus/derived/text/extracted.jsonl

python -m src.30_clean_text \
  --in corpus/derived/text/extracted.jsonl \
  --out corpus/derived/text/cleaned.jsonl

python -m src.40_chunk_text \
  --in corpus/derived/text/cleaned.jsonl \
  --out corpus/derived/chunks/chunks.jsonl

python -m src.60_embed_corpus \
  --in corpus/derived/chunks/chunks.jsonl \
  --out corpus/derived/embeddings

python -m src.90_main \
  --query "dynamic capabilities" \
  --k 5 \
  --enhance summarize
```

---

## Deliverables
- `README.md` (project overview, setup, examples)
- `ARCHITECTURE.md` (technical details)
- `TEAM_CONTRIBUTIONS.md` (who did what + peer-eval support)
- `requirements.txt`
- Source code under `src/`
- 2–3 minute demo video
- Peer assessment form (each member)
