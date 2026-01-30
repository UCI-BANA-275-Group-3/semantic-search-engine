# Next Steps: Cleaning Improvements Plan

## Goal
Improve `30_clean_text.py` (and upstream extraction if needed) to reduce structural noise, citation smog, and corrupted symbols, while preserving high-value content for embeddings and LLM use.

---

## Phase 0 — Baseline & Scoping (quick checks)
- [x] **Inventory artifacts**: sample 10–20 records from `extracted.jsonl` and `cleaned.jsonl`.
- [x] **Identify common boilerplate**: list repeated header/footer strings.
- [x] **Decide on Markdown path**: confirm if we adopt a layout-aware extractor (e.g., PyMuPDF4LLM/Marker/Docling).

Deliverable: short list of exact boilerplate strings + decision on Markdown conversion.

---

## Phase 1 — Tier 1 (High Impact Basics)
### 1A) Regex-based Boilerplate Removal
- [x] Implement a configurable list of known boilerplate strings.
- [x] Strip them from `cleaned_text`.

### 1B) Header/Footer Stripping
- [x] Use `pages` field (from extraction) to detect repeated first/last lines.
- [x] Remove those lines across pages when they match title/author/publication patterns.

Deliverable: fewer headers in mid-paragraph chunks and reduced noise.

---

## Phase 2 — Tier 2 (Structural Improvements)
### 2A) Markdown Conversion Integration
- [x] Evaluate **PyMuPDF4LLM** or **Marker/Docling** to extract Markdown.
- [x] If adopted, update `20_extract_text.py` to output Markdown and preserve headings.

### 2B) References/Bibliography Filtering
- [x] Detect “References” or “Bibliography” sections.
- [x] Move reference text into a separate field (e.g., `references_text`) and exclude from main cleaned text.

Deliverable: cleaner semantic text with references separated.

---

## Phase 3 — Tier 3 (Semantic Refinements)
### 3A) Citation Handling
- [x] Option A: normalize citations (e.g., replace `(Author 2018)` with `<CITATION>`).
- [ ] Option B: remove dense inline citations only when they occur in clusters.

### 3B) Unicode/Math Fixes
- [x] Map common extraction artifacts (e.g., `\u0007`, broken Greek letters).
- [x] Create a small replacement table applied during cleaning.

Deliverable: less corrupted math/notation and improved readability.

---

## Integration Points
- [x] `20_extract_text.py`: optional upgrade to Markdown-aware extraction.
- [x] `30_clean_text.py`: implement tiers 1–3 as configurable steps.
- [ ] `40_chunk_text.py`: leverage headers/markdown structure if available.

---

## Definition of Done (for Cleaning Update)
- [ ] Boilerplate removed and headers no longer appear mid-paragraph.
- [ ] References do not dominate retrieval results.
- [ ] Reduced citation noise in chunked output.
- [ ] No more common Unicode artifacts in cleaned text.
- [ ] Logs/metrics show reduced warnings and better text quality.

---

## Additional Next Steps (RAG-Ready Cleaning Improvements)

**Objective:** Refine `src/30_clean_text.py` to transform raw PDF text into high-quality, RAG-ready documents that maximize semantic density and minimize layout noise.

1) **Page Stitching & Flow Reconstruction**
- [x] Join the `pages` list into a single continuous `cleaned_text`.
- [x] Detect hyphenated line breaks across pages (e.g., `men-` + `tal`) and repair them.
- [x] Ensure a single space between Page N and Page N+1 unless a paragraph break is detected.

2) **Boilerplate & Header/Footer Removal**
- [ ] Strip static boilerplate (e.g., “You are reading copyrighted material…”).
- [ ] Strip dynamic headers (e.g., “90 Ajay Agrawal, Joshua Gans, and Avi Goldfarb”).
- [ ] Remove standalone page numbers at page boundaries.

Plan for 2) Boilerplate & Header/Footer Removal
1) **Static boilerplate list**: Expand `BOILERPLATE_PATTERNS` with corpus‑specific phrases (copyright notices, publisher notices, download banners).
2) **Dynamic header regexes**: Add regex rules for “page number + author list/title” patterns (e.g., `^\\d+\\s+Author`).
3) **Page‑aware cleaning**: If `pages` exists, apply header/footer removal before stitching pages to avoid carrying boilerplate into the body.
4) **Page‑number trimming**: Remove numeric‑only lines at top/bottom of each page (e.g., `^\\d+$` or `^\\d+\\s*$`).
5) **Logging/metrics**: Track counts of removed header/footer lines and boilerplate hits in the cleaning summary.

3) **Semantic Symbol Mapping**
- [ ] Replace Unicode artifacts with readable text or LaTeX.
- [ ] Example: map `\u0007` → `theta` (or `$\theta$`).
- [ ] Fix ligatures (e.g., `ﬁ` → `fi`, `ﬀ` → `ff`).

4) **Section-Aware Filtering (Bibliography)**
- [ ] Detect “References” or “Bibliography”.
- [ ] Option A: stop `cleaned_text` at the header.
- [ ] Option B: move the section into `references_text` for metadata only.

5) **Metadata Preservation**
- [ ] Move admin metadata (ISBNs, URLs, Volume Titles) into JSON metadata fields.
- [ ] Remove those from `cleaned_text` to save context window space.
