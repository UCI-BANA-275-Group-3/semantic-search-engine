import streamlit as st
from typing import List, Dict
import sys
import os

# Ensure the repository root is on sys.path so `src` imports work when
# running `streamlit run app/streamlit_app.py` (Streamlit may change CWD).
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.similarity_search import search_top_k
from src.llm_enhancement import summarize_top_k

st.set_page_config(page_title="Semantic Search", layout="wide")

st.title("Semantic Search — Local UI")

with st.form("search_form"):
    query = st.text_input("Enter your query", value="machine learning")
    k = st.number_input("Top K", min_value=1, max_value=50, value=5, step=1)
    run_search = st.form_submit_button("Search")

if run_search and query:
    st.info(f"Searching for: {query} (top {k})")
    try:
        results: List[Dict] = search_top_k(query=query, k=k)
    except Exception as e:
        st.error(f"Search failed: {e}")
        results = []

    if not results:
        st.warning("No results returned. Ensure embeddings exist (vectors.npy + meta.jsonl).")
    else:
        cols = st.columns([3, 1])
        with cols[0]:
            st.subheader("Top-K Results")
            for r in results:
                st.markdown(f"**{r.get('rank')}. {r.get('title') or '(no title)'}**  ")
                meta_line = f"_doc_id_: {r.get('doc_id')} — _score_: {r.get('score'):.4f}"
                st.caption(meta_line)
                snippet = (r.get('chunk_text') or r.get('text') or '')[:600].replace('\n', ' ')
                st.write(snippet)
                st.markdown("---")

        with cols[1]:
            st.subheader("Cleaned Summary (auto)")
            # `summarize_top_k` returns a deterministic, cleaned paragraph-style summary
            summary = summarize_top_k(query, results)
            st.write(summary)
            st.download_button("Download summary (TXT)", data=summary, file_name="summary.txt")

        # Optional: allow download of results
        st.download_button("Download results (JSON)", data=str(results), file_name="topk_results.json")
