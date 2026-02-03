"""
Unit tests for semantic search pipeline.

Run with: pytest tests/test_core.py
"""

import json
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

# Import modules to test
from src.embedding_backends import _l2_normalize
from src.similarity_search import topk_similarities, _maybe_normalize_rows


class TestEmbeddingBackends:
    """Test embedding backend utilities."""
    
    def test_l2_normalize_1d(self):
        """Test L2 normalization of 1D vector."""
        x = np.array([3.0, 4.0], dtype=np.float32)
        normalized = _l2_normalize(x)
        
        # Norm should be 1
        norm = np.linalg.norm(normalized)
        assert np.isclose(norm, 1.0), f"Expected norm 1.0, got {norm}"
        
        # Should be 3/5, 4/5
        expected = np.array([0.6, 0.8], dtype=np.float32)
        assert np.allclose(normalized, expected), f"Expected {expected}, got {normalized}"
    
    def test_l2_normalize_2d(self):
        """Test L2 normalization of 2D array (row-wise)."""
        x = np.array(
            [[3.0, 4.0], [0.0, 5.0]],
            dtype=np.float32,
        )
        normalized = _l2_normalize(x)
        
        # Each row should have norm 1
        norms = np.linalg.norm(normalized, axis=1)
        assert np.allclose(norms, 1.0), f"Expected norms [1, 1], got {norms}"
    
    def test_l2_normalize_zero_vector(self):
        """Test L2 normalization with near-zero vector (eps safety)."""
        x = np.array([1e-15, 1e-15], dtype=np.float32)
        normalized = _l2_normalize(x, eps=1e-12)
        
        # Should not raise, result should be valid
        assert not np.any(np.isnan(normalized)), "Normalization produced NaN"
        assert not np.any(np.isinf(normalized)), "Normalization produced inf"


class TestSimilaritySearch:
    """Test similarity search core functions."""
    
    def test_topk_similarities_basic(self):
        """Test top-K selection with simple vectors."""
        # 3 vectors, 2D
        vectors = np.array([
            [1.0, 0.0],
            [0.8, 0.6],
            [0.0, 1.0],
        ], dtype=np.float32)
        
        query = np.array([1.0, 0.0], dtype=np.float32)
        k = 2
        
        idx = topk_similarities(query, vectors, k)
        
        # Should return indices of top-2 most similar
        # Assuming vectors are normalized:
        # sim[0] = 1.0 (identical)
        # sim[1] = 0.8 (dot product)
        # sim[2] = 0.0 (orthogonal)
        # So top-2 should be [0, 1]
        
        assert len(idx) == 2, f"Expected 2 results, got {len(idx)}"
        assert idx[0] == 0, f"First result should be index 0, got {idx[0]}"
    
    def test_topk_similarities_k_larger_than_n(self):
        """Test top-K when K > number of vectors."""
        vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        query = np.array([1.0, 0.0], dtype=np.float32)
        k = 10  # More than available vectors
        
        idx = topk_similarities(query, vectors, k)
        
        # Should return all vectors (capped at N)
        assert len(idx) == 2, f"Expected 2 results (N), got {len(idx)}"
    
    def test_topk_similarities_dimension_mismatch(self):
        """Test that dimension mismatch raises error."""
        vectors = np.array([[1.0, 0.0]], dtype=np.float32)
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # 3D query
        
        with pytest.raises(SystemExit):
            topk_similarities(query, vectors, k=1)
    
    def test_maybe_normalize_rows(self):
        """Test row-wise normalization."""
        x = np.array([
            [3.0, 4.0],
            [0.0, 5.0],
        ], dtype=np.float32)
        
        normalized = _maybe_normalize_rows(x)
        
        # Check norms
        norms = np.linalg.norm(normalized, axis=1)
        assert np.allclose(norms, 1.0), f"Expected unit norms, got {norms}"


class TestIOFunctions:
    """Test I/O utilities."""
    
    def test_read_write_jsonl_roundtrip(self):
        """Test JSONL read/write roundtrip."""
        from src.embed_corpus import read_jsonl, write_jsonl
        
        records = [
            {"id": 1, "text": "hello"},
            {"id": 2, "text": "world"},
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonl"
            
            # Write
            write_jsonl(str(path), records)
            assert path.exists(), "File not created"
            
            # Read
            read_records = list(read_jsonl(str(path)))
            
            assert len(read_records) == 2, f"Expected 2 records, got {len(read_records)}"
            assert read_records[0]["id"] == 1
            assert read_records[1]["text"] == "world"
    
    def test_read_jsonl_with_empty_lines(self):
        """Test JSONL reading with blank lines."""
        from src.embed_corpus import read_jsonl
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonl"
            
            # Write JSONL with empty lines
            with open(path, "w") as f:
                f.write('{"id": 1}\n')
                f.write('\n')  # Empty line
                f.write('{"id": 2}\n')
            
            # Read should skip empty lines
            records = list(read_jsonl(str(path)))
            assert len(records) == 2, f"Expected 2 records (empty line skipped), got {len(records)}"
    
    def test_read_jsonl_invalid_json(self):
        """Test JSONL reading with invalid JSON."""
        from src.embed_corpus import read_jsonl
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonl"
            
            # Write invalid JSON
            with open(path, "w") as f:
                f.write('{"id": 1}\n')
                f.write('invalid json\n')
            
            # Should raise SystemExit
            with pytest.raises(SystemExit):
                list(read_jsonl(str(path)))


class TestTextCleaning:
    """Test text cleaning functions."""
    
    def test_unicode_replacements(self):
        """Test that unicode characters are replaced correctly."""
        from src.30_clean_text import UNICODE_FIXES
        
        # Check some common replacements exist
        assert UNICODE_FIXES.get('\u03B1') == '$\\alpha$', "Greek alpha not mapped"
        assert UNICODE_FIXES.get('\u03C0') == '$\\pi$', "Greek pi not mapped"
        assert UNICODE_FIXES.get('"') == '"', "Smart quote not mapped"


class TestChunking:
    """Test text chunking functions."""
    
    def test_chunk_determinism(self):
        """Test that chunking is deterministic."""
        from src.chunk_text import chunk_text_deterministic
        
        text = "Sentence one. " * 100  # Long text
        
        chunks1 = chunk_text_deterministic(text, max_tokens=50)
        chunks2 = chunk_text_deterministic(text, max_tokens=50)
        
        # Same text should produce same chunks
        assert len(chunks1) == len(chunks2), "Chunk count inconsistent"
        assert chunks1 == chunks2, "Chunks differ on second run"


# Integration tests
class TestPipelineIntegration:
    """Integration tests for the full pipeline."""
    
    def test_end_to_end_small_corpus(self):
        """
        Test that a small corpus can be processed end-to-end.
        
        This is a skeleton test; actual implementation would need
        a small test corpus under tests/fixtures/
        """
        pytest.skip("Requires test corpus fixtures to be implemented")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
