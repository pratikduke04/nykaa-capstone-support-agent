"""Shared constants for the RAG layer. Pure Python - no heavy imports here."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KB_DIR = PROJECT_ROOT / "kb"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

EMBED_MODEL = "all-MiniLM-L6-v2"      # small, free, runs locally on CPU

COLLECTION_FIXED = "kb_fixed"          # fixed-size-with-overlap chunks
COLLECTION_SENTENCE = "kb_sentence"    # one-sentence chunks

# Fixed-size chunking parameters (measured in WORDS so we never cut mid-word)
FIXED_CHUNK_WORDS = 40
FIXED_OVERLAP_WORDS = 10