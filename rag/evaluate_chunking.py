"""
Task 5 - compare the two chunking strategies at the DOCUMENT level.

For each demo query and each collection:
  1. retrieve the top-k CHUNKS (same k for both collections, no threshold applied)
  2. map each chunk back to its parent document and DEDUPLICATE (keep rank order)
  3. precision = relevant docs retrieved / unique docs retrieved
     recall    = relevant docs retrieved / total relevant docs
The per-query arithmetic is printed so every number can be checked by hand.

Run:  python -m rag.evaluate_chunking
"""
from rag.config import COLLECTION_FIXED, COLLECTION_SENTENCE, TOP_K


def short(doc_id: str) -> str:
    return doc_id.split("_")[0]          # '01_return_window_by_category' -> '01'


def dedup_docs(chunks: list[dict]) -> list[str]:
    """Parent doc_ids of the chunks, duplicates removed, rank order preserved."""
    seen = []
    for c in chunks:
        if c["doc_id"] not in seen:
            seen.append(c["doc_id"])
    return seen


def precision_recall(retrieved_docs: list[str], relevant: set[str]):
    hits = [d for d in retrieved_docs if d in relevant]
    precision = len(hits) / len(retrieved_docs) if retrieved_docs else 0.0
    recall = len(hits) / len(relevant) if relevant else 0.0
    return precision, recall, hits


def main():
    from rag.grounded import retrieve          # imported here so the maths above is testable alone
    from rag.queries import DEMO_QUERIES

    totals = {COLLECTION_FIXED: [], COLLECTION_SENTENCE: []}
    for i, item in enumerate(DEMO_QUERIES, 1):
        q, relevant = item["q"], item["relevant"]
        print("=" * 90)
        print(f"Q{i}: {q}")
        print(f"    relevant docs = {sorted(short(d) for d in relevant)}  (n = {len(relevant)})")
        for name in (COLLECTION_FIXED, COLLECTION_SENTENCE):
            chunks = retrieve(q, name, k=TOP_K)
            docs = dedup_docs(chunks)
            p, r, hits = precision_recall(docs, relevant)
            totals[name].append((p, r))
            print(f"  [{name}]")
            print(f"    top-{TOP_K} chunks -> docs : {[short(c['doc_id']) for c in chunks]}")
            print(f"    unique docs after dedup   : {[short(d) for d in docs]}  (u = {len(docs)})")
            print(f"    relevant hits             : {[short(d) for d in hits]}  (h = {len(hits)})")
            print(f"    precision = h/u = {len(hits)}/{len(docs)} = {p:.3f}")
            print(f"    recall    = h/n = {len(hits)}/{len(relevant)} = {r:.3f}")

    print("\n" + "=" * 90)
    print(f"SUMMARY over {len(DEMO_QUERIES)} queries (top-{TOP_K} chunks, macro-average)")
    print("=" * 90)
    print(f"{'collection':<14}{'mean precision':>16}{'mean recall':>14}")
    for name, rows in totals.items():
        n = len(rows)
        mp = sum(p for p, _ in rows) / n
        mr = sum(r for _, r in rows) / n
        ps = " + ".join(f"{p:.3f}" for p, _ in rows)
        rs = " + ".join(f"{r:.3f}" for _, r in rows)
        print(f"{name:<14}{mp:>16.3f}{mr:>14.3f}")
        print(f"   precision = ({ps}) / {n}")
        print(f"   recall    = ({rs}) / {n}")


if __name__ == "__main__":
    main()