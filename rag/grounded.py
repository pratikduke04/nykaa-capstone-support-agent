"""
Task 4 - grounded generation.

retrieve()      : top-k chunks from one collection, with cosine similarity.
answer_query()  : if the top-1 similarity is below the CALIBRATED threshold -> fallback
                  ("I don't know"); otherwise answer using ONLY the retrieved chunks.

MOCK_LLM: the generator is deterministic and extractive - it can only quote the retrieved
context, so under MOCK_LLM an answer is grounded by construction. Similarity is the only
signal we have for "does the context support the question?", hence the calibrated threshold.
"""
from rag.config import COLLECTION_SENTENCE, COLLECTION_FIXED, SIMILARITY_THRESHOLD, TOP_K
from rag.index import embed, get_client, get_collection

FALLBACK_ANSWER = ("I don't have enough information in the Nykaa policy knowledge base "
                   "to answer that. Please contact customer support.")

_client = None


def _col(name):
    global _client
    if _client is None:
        _client = get_client()
    return get_collection(_client, name)


def retrieve(query: str, collection_name: str = COLLECTION_SENTENCE, k: int = TOP_K) -> list[dict]:
    res = _col(collection_name).query(query_embeddings=embed([query]), n_results=k)
    return [
        {"chunk_id": cid, "doc_id": meta["doc_id"], "text": text, "similarity": 1 - dist}
        for cid, meta, text, dist in zip(res["ids"][0], res["metadatas"][0],
                                         res["documents"][0], res["distances"][0])
    ]


def get_threshold(collection_name: str) -> float:
    t = SIMILARITY_THRESHOLD[collection_name]
    if t is None:
        raise RuntimeError("Threshold not set. Run `python -m rag.calibrate`, then paste the "
                           "measured value into SIMILARITY_THRESHOLD in rag/config.py")
    return t


def mock_generate(query: str, context_chunks: list[dict]) -> str:
    """Deterministic extractive 'LLM': answers only by quoting the retrieved context."""
    return "According to our policy: " + " ".join(c["text"] for c in context_chunks)


def answer_query(query: str, collection_name: str = COLLECTION_SENTENCE, k: int = TOP_K) -> dict:
    chunks = retrieve(query, collection_name, k)
    threshold = get_threshold(collection_name)
    top_sim = chunks[0]["similarity"] if chunks else 0.0
    if top_sim < threshold:
        return {"query": query, "answer": FALLBACK_ANSWER, "fallback": True,
                "top_similarity": top_sim, "threshold": threshold, "sources": []}
    context = [c for c in chunks if c["similarity"] >= threshold]
    return {"query": query, "answer": mock_generate(query, context), "fallback": False,
            "top_similarity": top_sim, "threshold": threshold,
            "sources": [c["chunk_id"] for c in context]}


if __name__ == "__main__":
    from rag.queries import DEMO_QUERIES, DEMO_OUT_OF_SCOPE
    for name in (COLLECTION_SENTENCE, COLLECTION_FIXED):
        print("=" * 90)
        print(f"COLLECTION: {name}   threshold = {get_threshold(name)}")
        print("=" * 90)
        queries = [d["q"] for d in DEMO_QUERIES] + [DEMO_OUT_OF_SCOPE]
        for q in queries:
            r = answer_query(q, name)
            tag = "FALLBACK (I don't know)" if r["fallback"] else "ANSWERED"
            print(f"\nQ: {q}\n  top-1 sim = {r['top_similarity']:.3f}  -> {tag}")
            print(f"  A: {r['answer']}")
            if r["sources"]:
                print(f"  sources: {r['sources']}")