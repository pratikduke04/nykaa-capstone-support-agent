"""
Task 3 - load the knowledge base and split it with two chunking strategies.

  1. fixed-size with overlap : windows of FIXED_CHUNK_WORDS words, stepping forward
                               by (chunk - overlap) words, so neighbouring chunks share text.
  2. sentence-based          : one chunk per sentence.

Every chunk keeps its parent doc_id so Task 5 can map chunks back to documents.
"""
import re
from rag.config import KB_DIR, FIXED_CHUNK_WORDS, FIXED_OVERLAP_WORDS


def load_documents(kb_dir=KB_DIR) -> list[dict]:
    """Read every .md file. First line '# Title' is the title; the rest is the body."""
    docs = []
    for path in sorted(kb_dir.glob("*.md")):
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        title = lines[0].lstrip("# ").strip()
        body = " ".join(line.strip() for line in lines[1:] if line.strip())
        docs.append({"doc_id": path.stem, "title": title, "text": body})
    return docs


def chunk_fixed(doc: dict, size: int = FIXED_CHUNK_WORDS, overlap: int = FIXED_OVERLAP_WORDS) -> list[dict]:
    assert 0 <= overlap < size, "overlap must be smaller than chunk size"
    words = doc["text"].split()
    step = size - overlap
    chunks = []
    for i, start in enumerate(range(0, len(words), step)):
        window = words[start:start + size]
        chunks.append({
            "chunk_id": f"{doc['doc_id']}::fixed::{i}",
            "doc_id": doc["doc_id"],
            "title": doc["title"],
            "strategy": "fixed",
            "text": " ".join(window),
        })
        if start + size >= len(words):   # this window already reached the end
            break
    return chunks


def chunk_sentences(doc: dict) -> list[dict]:
    sentences = re.split(r"(?<=[.!?])\s+", doc["text"])
    return [{
        "chunk_id": f"{doc['doc_id']}::sent::{i}",
        "doc_id": doc["doc_id"],
        "title": doc["title"],
        "strategy": "sentence",
        "text": s.strip(),
    } for i, s in enumerate(sentences) if s.strip()]


def build_all_chunks() -> tuple[list[dict], list[dict]]:
    docs = load_documents()
    fixed = [c for d in docs for c in chunk_fixed(d)]
    sent = [c for d in docs for c in chunk_sentences(d)]
    return fixed, sent


if __name__ == "__main__":
    docs = load_documents()
    fixed, sent = build_all_chunks()
    print(f"Documents: {len(docs)}")
    print(f"Fixed-size chunks ({FIXED_CHUNK_WORDS} words, {FIXED_OVERLAP_WORDS} overlap): {len(fixed)}")
    print(f"Sentence chunks: {len(sent)}")
    d = docs[0]
    print(f"\n--- Example document: {d['doc_id']} ---")
    print("\nFixed chunks:")
    for c in chunk_fixed(d):
        print(f"  [{c['chunk_id']}] {c['text']}")
    print("\nSentence chunks:")
    for c in chunk_sentences(d):
        print(f"  [{c['chunk_id']}] {c['text']}")