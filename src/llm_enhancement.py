from typing import List, Dict


def _build_context(results: List[Dict], max_chars_per_chunk: int = 400) -> str:
    """Build a text context from top-K results."""
    lines = []
    for r in results:
        label = (
            str(r.get("doc_id"))
            or str(r.get("docid"))
            or str(r.get("pdffile"))
            or str(r.get("chunk_id"))
            or str(r.get("chunkid"))
            or str(r.get("rank"))
            or "source"
        )

        text = (
            r.get("chunk_text")
            or r.get("text")
            or r.get("preview")
            or r.get("snippet")
            or ""
        )

        if not text:
            continue

        text = str(text).strip()
        if len(text) > max_chars_per_chunk:
            text = text[:max_chars_per_chunk] + "..."

        lines.append(f"[{label}] {text}")

    return "\n\n".join(lines)


def summarize_top_k(query: str, results: List[Dict]) -> str:
    """Deterministic top-K summarizer (no external LLMs)."""
    if not results:
        return "No results were retrieved for your query."

    # Enhanced: Order sentences by relevance, filter boilerplate, add connective text
    import re
    seen = set()
    summary_sentences = []
    query_lc = query.lower()
    boilerplate_patterns = [
        r"copyright",
        r"all rights reserved",
        r"u\.s\. copyright law",
        r"for helpful comments",
        r"responsibility for all errors",
        r"thank[s]? to",
        r"see http",
        r"this pdf is a selection",
        r"no results were retrieved",
        r"no content found",
    ]
    def is_boilerplate(s: str) -> bool:
        s_lc = s.lower()
        return any(re.search(pat, s_lc) for pat in boilerplate_patterns)

    def sent_score(s: str, sents: list) -> int:
        # Higher score for query overlap, length, and position (topic sentence)
        score = 0
        s_lc = s.lower()
        if query_lc in s_lc:
            score += 3
        score += min(len(s.split()), 25) // 5  # up to +5 for length
        if s == sents[0]:
            score += 2  # topic sentence
        return score

    # Collect candidate sentences from top chunks, score them, then pick top 3-5
    candidates: List[tuple] = []  # (score, sentence)
    for r in results[:8]:
        text = (
            r.get("chunk_text")
            or r.get("text")
            or r.get("preview")
            or r.get("snippet")
            or ""
        )
        text = str(text).strip()
        if not text:
            continue
        sents = re.split(r'(?<=[.!?])\s+', text)
        for pos, s in enumerate(sents):
            s_str = s.strip()
            if not s_str:
                continue
            s_lc = s_str.lower()
            if s_lc in seen:
                continue
            if is_boilerplate(s_str):
                continue
            if len(s_str) < 20 and query_lc not in s_lc:
                continue
            # normalize
            s_clean = re.sub(r"^[\W\d]+", "", s_str).replace("\n", " ").strip()
            if not s_clean:
                continue
            score = sent_score(s_clean, sents)
            # small bonus for earlier sentences in chunk
            score += max(0, 2 - pos)
            candidates.append((score, s_clean))
            seen.add(s_lc)

    if not candidates:
        return "No content found in the retrieved results."

    # Pick top candidates, dedupe by lowercase, limit to 3-5 sentences
    candidates = sorted(candidates, key=lambda x: x[0], reverse=True)
    selected: List[str] = []
    sel_set = set()
    for score, sent in candidates:
        key = sent.lower()
        if key in sel_set:
            continue
        selected.append(sent)
        sel_set.add(key)
        if len(selected) >= 4:
            break

    # Ensure sentences end with proper punctuation
    def ensure_punct(s: str) -> str:
        s = s.strip()
        if not s:
            return s
        if s[-1] not in '.!?':
            return s + '.'
        return s

    selected = [ensure_punct(s) for s in selected]
    paragraph = ' '.join(selected)
    return f"Based on your query '{query}', {paragraph}"


# Future: add LLM-based enhancements (QA, compare, aggregate) as optional modules.
