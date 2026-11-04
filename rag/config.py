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

# ---- Task 4: retrieval + calibrated "I don't know" threshold ----
TOP_K = 3

# Top-1 cosine-similarity cut-off PER COLLECTION (the two collections have different
# similarity scales). Leave as None until you have run:  python -m rag.calibrate
# then paste the measured values here.
SIMILARITY_THRESHOLD = {
    COLLECTION_FIXED: 0.27,
    COLLECTION_SENTENCE: 0.26,
}

# Answer-quality filter used by the crew's RAG tool (NOT the fallback threshold):
# only keep retrieved documents whose best chunk is within this margin of the top-1 chunk.
RELATIVE_MARGIN = 0.30