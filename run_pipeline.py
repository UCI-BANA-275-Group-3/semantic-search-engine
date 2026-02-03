#!/usr/bin/env python3
"""
Master orchestration script for the semantic search pipeline.

Runs all stages sequentially with error handling, logging, and progress tracking.

Usage:
  python run_pipeline.py [--corpus-path CORPUS_PATH] [--skip-stages STAGE1,STAGE2,...]

Example:
  python run_pipeline.py --corpus-path corpus --skip-stages validate  # Skip validation
  python run_pipeline.py  # Run all stages with defaults
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Manages the complete semantic search pipeline."""
    
    def __init__(self, corpus_path: str = "corpus", skip_stages: Optional[List[str]] = None):
        """
        Initialize the orchestrator.
        
        Args:
            corpus_path: Base path to corpus directory
            skip_stages: List of stage names to skip (e.g., ['validate', 'embed'])
        """
        self.corpus_path = Path(corpus_path)
        self.skip_stages = set(skip_stages or [])
        self.results = {}
        self.start_time = datetime.now()
        
        # Ensure directories exist
        self._ensure_directories()
    
    def _ensure_directories(self) -> None:
        """Create required directories."""
        dirs = [
            self.corpus_path / "raw" / "zotero" / "metadata",
            self.corpus_path / "raw" / "zotero" / "storage",
            self.corpus_path / "derived" / "manifest",
            self.corpus_path / "derived" / "text",
            self.corpus_path / "derived" / "embeddings",
            self.corpus_path / "logs",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {d}")
    
    def _run_stage(
        self,
        stage_name: str,
        stage_id: str,
        description: str,
        command: List[str],
        required_inputs: Optional[List[Path]] = None,
    ) -> bool:
        """
        Run a single pipeline stage.
        
        Args:
            stage_name: Human-readable stage name
            stage_id: Short ID (e.g., 'manifest', 'validate')
            description: Longer description
            command: Command to execute
            required_inputs: List of files that must exist before running
        
        Returns:
            True if successful, False otherwise
        """
        if stage_id in self.skip_stages:
            logger.info(f"⊘ Skipping stage {stage_id}: {description}")
            return True
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Stage {stage_id}: {stage_name}")
        logger.info(f"{'='*80}")
        logger.info(description)
        
        # Check required inputs
        if required_inputs:
            missing = [p for p in required_inputs if not p.exists()]
            if missing:
                logger.error(f"Missing required input files: {missing}")
                return False
        
        # Run command
        logger.info(f"Running: {' '.join(command)}")
        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=False,
                text=True,
            )
            logger.info(f"✓ Stage {stage_id} completed successfully")
            self.results[stage_id] = {"status": "success", "return_code": result.returncode}
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Stage {stage_id} failed with return code {e.returncode}")
            logger.error(f"Error output: {e.stderr}")
            self.results[stage_id] = {"status": "failed", "return_code": e.returncode}
            return False
        except FileNotFoundError:
            logger.error(f"✗ Stage {stage_id}: Command not found. Check Python path.")
            self.results[stage_id] = {"status": "failed", "error": "Command not found"}
            return False
        except Exception as e:
            logger.error(f"✗ Stage {stage_id} failed with exception: {e}")
            self.results[stage_id] = {"status": "failed", "error": str(e)}
            return False
    
    def run_full_pipeline(self) -> bool:
        """
        Execute the complete pipeline.
        
        Returns:
            True if all stages succeeded, False otherwise
        """
        logger.info(f"Starting semantic search pipeline at {self.start_time}")
        logger.info(f"Corpus path: {self.corpus_path}")
        logger.info(f"Skipping stages: {self.skip_stages or 'none'}")
        
        stages = [
            (
                "00: Build Manifest",
                "manifest",
                "Extract document metadata from Zotero library.json",
                [
                    sys.executable, "-m", "src.00_build_manifest",
                    "--metadata", str(self.corpus_path / "raw" / "zotero" / "metadata" / "library.json"),
                    "--storage-root", str(self.corpus_path / "raw" / "zotero" / "storage"),
                    "--out", str(self.corpus_path / "derived" / "manifest" / "manifest.jsonl"),
                ],
                [self.corpus_path / "raw" / "zotero" / "metadata" / "library.json"],
            ),
            (
                "10: Validate Corpus",
                "validate",
                "Validate manifest and check file integrity",
                [
                    sys.executable, "-m", "src.10_validate_corpus",
                    "--manifest", str(self.corpus_path / "derived" / "manifest" / "manifest.jsonl"),
                    "--logs", str(self.corpus_path / "logs"),
                ],
                [self.corpus_path / "derived" / "manifest" / "manifest.jsonl"],
            ),
            (
                "20: Extract Text",
                "extract",
                "Extract raw text from PDFs, HTML, and text files",
                [
                    sys.executable, "-m", "src.20_extract_text",
                    "--manifest", str(self.corpus_path / "derived" / "manifest" / "manifest.jsonl"),
                    "--out", str(self.corpus_path / "derived" / "text" / "extracted.jsonl"),
                    "--logs", str(self.corpus_path / "logs"),
                ],
                [self.corpus_path / "derived" / "manifest" / "manifest.jsonl"],
            ),
            (
                "30: Clean Text",
                "clean",
                "Normalize text and fix encoding artifacts",
                [
                    sys.executable, "-m", "src.30_clean_text",
                    "--in", str(self.corpus_path / "derived" / "text" / "extracted.jsonl"),
                    "--out", str(self.corpus_path / "derived" / "text" / "cleaned.jsonl"),
                    "--logs", str(self.corpus_path / "logs"),
                ],
                [self.corpus_path / "derived" / "text" / "extracted.jsonl"],
            ),
            (
                "40: Chunk Text",
                "chunk",
                "Split documents into token-aware chunks with overlap",
                [
                    sys.executable, "-m", "src.chunk_text",
                    "--in", str(self.corpus_path / "derived" / "text" / "cleaned.jsonl"),
                    "--out", str(self.corpus_path / "derived" / "text" / "chunks.jsonl"),
                    "--logs", str(self.corpus_path / "logs"),
                ],
                [self.corpus_path / "derived" / "text" / "cleaned.jsonl"],
            ),
            (
                "50: Embed Corpus",
                "embed",
                "Convert chunks to dense vectors using SentenceTransformers",
                [
                    sys.executable, "-m", "src.embed_corpus",
                    "--in", str(self.corpus_path / "derived" / "text" / "chunks.jsonl"),
                    "--out-dir", str(self.corpus_path / "derived" / "embeddings"),
                ],
                [self.corpus_path / "derived" / "text" / "chunks.jsonl"],
            ),
        ]
        
        # Execute each stage
        failed_stages = []
        for stage_name, stage_id, description, command, required_inputs in stages:
            if not self._run_stage(stage_name, stage_id, description, command, required_inputs):
                failed_stages.append(stage_id)
                logger.warning(f"Pipeline halted due to failed stage: {stage_id}")
                break  # Stop on first failure
        
        # Summary
        elapsed = datetime.now() - self.start_time
        logger.info(f"\n{'='*80}")
        logger.info(f"Pipeline Summary")
        logger.info(f"{'='*80}")
        logger.info(f"Completed stages: {len([s for s in self.results.values() if s['status'] == 'success'])}")
        logger.info(f"Failed stages: {len([s for s in self.results.values() if s['status'] == 'failed'])}")
        logger.info(f"Elapsed time: {elapsed}")
        
        if failed_stages:
            logger.error(f"\n✗ Pipeline FAILED at stage(s): {', '.join(failed_stages)}")
            return False
        else:
            logger.info("\n✓ Pipeline COMPLETED successfully")
            logger.info(f"\nNext steps:")
            logger.info(f"  1. Run similarity search: python -m src.similarity_search --query 'your query'")
            logger.info(f"  2. Try the CLI: python -m src.90_main --query 'your query' --topk-file topk.jsonl")
            return True
    
    def save_results(self, output_file: str = "pipeline_results.json") -> None:
        """Save pipeline results to a JSON file."""
        results_dict = {
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "elapsed_seconds": (datetime.now() - self.start_time).total_seconds(),
            "stages": self.results,
        }
        with open(output_file, "w") as f:
            json.dump(results_dict, f, indent=2)
        logger.info(f"Results saved to: {output_file}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Master orchestration script for the semantic search pipeline",
        epilog=(
            "Example: python run_pipeline.py --corpus-path corpus --skip-stages validate\n"
            "This will run all stages except validation."
        ),
    )
    parser.add_argument(
        "--corpus-path",
        default="corpus",
        help="Path to corpus directory (default: corpus)",
    )
    parser.add_argument(
        "--skip-stages",
        help="Comma-separated list of stages to skip (e.g., 'validate,embed')",
    )
    parser.add_argument(
        "--save-results",
        type=str,
        default="pipeline_results.json",
        help="Path to save results JSON (default: pipeline_results.json)",
    )
    
    args = parser.parse_args()
    skip_stages = args.skip_stages.split(",") if args.skip_stages else []
    
    try:
        orchestrator = PipelineOrchestrator(
            corpus_path=args.corpus_path,
            skip_stages=skip_stages,
        )
        success = orchestrator.run_full_pipeline()
        orchestrator.save_results(args.save_results)
        
        return 0 if success else 1
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
