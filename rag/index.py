"""
Task 3 - embed every chunk with SentenceTransformers and index each strategy
in its OWN ChromaDB collection using collection.upsert().

Run from the project root:   python -m rag.index
"""
import chromadb
from sentence_transformers import SentenceTransformer

from rag.chunking import build_all_chunks
from rag.config import CHROMA_DIR, EMBED_MODEL, COLLECTION_FIXED, COLLECTION_SENTENCE

_model = None


def get_model() -> SentenceTransformer:
    """Load the embedding model once and reuse it (loading takes a few seconds)."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def embed(texts: list[str]) -> list[list[float]]:
    # normalize_embeddings=True -> unit vectors, so cosine similarity is well-behaved
    return get_model().encode(texts, normalize_embeddings=True).tolist()


def get_client():
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection(client, name: str):
    # cosine distance: similarity = 1 - distance (used for the threshold in Task 4)
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


def index_chunks(client, collection_name: str, chunks: list[dict]):
    col = get_collection(client, collection_name)
    col.upsert(
        ids=[c["chunk_id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        embeddings=embed([c["text"] for c in chunks]),
        metadatas=[{"doc_id": c["doc_id"], "title": c["title"], "strategy": c["strategy"]} for c in chunks],
    )
    return col


def show_sample_query(client, query: str, k: int = 3):
    q_emb = embed([query])
    for name in (COLLECTION_FIXED, COLLECTION_SENTENCE):
        col = get_collection(client, name)
        res = col.query(query_embeddings=q_emb, n_results=k)
        print(f"\n[{name}] top-{k} for: {query!r}")
        for cid, text, dist in zip(res["ids"][0], res["documents"][0], res["distances"][0]):
            print(f"  sim={1 - dist:.3f}  {cid}\n      {text}")


if __name__ == "__main__":
    fixed, sent = build_all_chunks()
    client = get_client()
    index_chunks(client, COLLECTION_FIXED, fixed)
    index_chunks(client, COLLECTION_SENTENCE, sent)
    print(f"Embedding model : {EMBED_MODEL}")
    print(f"{COLLECTION_FIXED:<14}: {get_collection(client, COLLECTION_FIXED).count()} chunks indexed")
    print(f"{COLLECTION_SENTENCE:<14}: {get_collection(client, COLLECTION_SENTENCE).count()} chunks indexed")
    show_sample_query(client, "How many days do I have to return a lipstick?")