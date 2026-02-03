"""Root shim for backward compatibility.

This file delegates to `src.llm_enhancement` so older imports (top-level)
continue to work after we consolidate code under `src/`.
"""

from src import llm_enhancement as _src_llm

summarize_top_k = _src_llm.summarize_top_k
_build_context = _src_llm._build_context

__all__ = ["summarize_top_k", "_build_context"]

if __name__ == "__main__":
    print("This is a compatibility shim. Import src.llm_enhancement.summarize_top_k.")
