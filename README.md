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

1) **Manifest building**
- Script: `src/00_build_manifest.py`
- Inputs: `corpus/raw/zotero/metadata/library.json`, `corpus/raw/zotero/storage/**`
- Output: `corpus/derived/manifest/manifest.jsonl`

2) **Corpus validation**
- Script: `src/10_validate_corpus.py`
- Inputs: manifest + raw storage
- Outputs: validation logs (location TBD)

3) **Text extraction**
- Script: `src/20_extract_text.py`
- Input: `corpus/derived/manifest/manifest.jsonl`
- Output: `corpus/derived/text/extracted.jsonl`

4) **Cleaning**
- Script: `src/30_clean_text.py`
- Input: `corpus/derived/text/extracted.jsonl`
- Output: `corpus/derived/text/cleaned.jsonl`

5) **Chunking**
- Script: `src/40_chunk_text.py`
- Input: `corpus/derived/text/cleaned.jsonl`
- Output: `corpus/derived/chunks/chunks.jsonl`

6) **Embeddings**
- Scripts: `src/50_embedding_backends.py`, `src/60_embed_corpus.py`
- Input: `corpus/derived/chunks/chunks.jsonl`
- Outputs: `corpus/derived/embeddings/embeddings.npy`, `corpus/derived/embeddings/index.jsonl`

7) **Similarity search**
- Script: `src/70_similarity_search.py`
- Inputs: embeddings artifacts
- Output: top-K chunks with cosine scores

8) **LLM enhancement (choose ONE)**
- Script: `src/80_llm_enhancement.py`
- Modes: summarize / QA / compare

9) **CLI interface**
- Script: `src/90_main.py`
- Runs retrieval + optional enhancement

---

## Setup

### Python
- Recommended: Python 3.10+

### Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

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
