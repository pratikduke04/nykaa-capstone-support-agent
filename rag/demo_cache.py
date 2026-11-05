"""Task 16 demo: before / after evidence for the response cache.   Run: python -m rag.demo_cache"""
import time

import rag.grounded as grounded
from rag import cache
from rag.config import COLLECTION_SENTENCE

# ---- count how often the expensive step really runs: wrap the embedding function ----
EMBED_CALLS = {"n": 0}
_real_embed = grounded.embed


def counting_embed(texts):
    EMBED_CALLS["n"] += 1
    return _real_embed(texts)


grounded.embed = counting_embed

WORKLOAD = [                                   # 12 requests, 4 distinct questions, mixed case and spacing
    "What is the return window for apparel?",
    "How long does a COD refund take?",
    "what is the return window for apparel?",
    "Can I cancel my order after it has shipped?",
    "HOW LONG DOES A COD REFUND TAKE?",
    "What is the return window for apparel?",
    "  Can I cancel my order   after it has shipped?  ",
    "WHAT IS THE RETURN WINDOW FOR APPAREL?",
    "Do you ship to Mars?",
    "do you ship to mars?",
    "How long does a COD refund take?",
    "can i cancel my order after it has shipped?",
]


def run(fn):
    rows, t_all = [], time.perf_counter()
    for q in WORKLOAD:
        t0 = time.perf_counter()
        r = fn(q)
        rows.append((q, r, (time.perf_counter() - t0) * 1000))
    return rows, (time.perf_counter() - t_all) * 1000


grounded.embed(["warm up the embedding model"])          # load the model once so timings are fair
EMBED_CALLS["n"] = 0

print("=== BEFORE: answer_query() called directly (no cache) ===")
rows_a, total_a = run(lambda q: grounded.answer_query(q, COLLECTION_SENTENCE))
embeds_a = EMBED_CALLS["n"]
for q, r, ms in rows_a:
    print(f"  {ms:7.1f} ms  fallback={r['fallback']!s:<5}  {q.strip()!r}")
print(f"  -> {len(WORKLOAD)} requests, {embeds_a} embedding+search+generation runs, {total_a:.0f} ms total")

print("\n=== AFTER: cached_answer_query() (cache keyed by normalized query) ===")
cache.clear(); cache.reset_stats(); EMBED_CALLS["n"] = 0
rows_b, total_b = run(lambda q: cache.cached_answer_query(q, COLLECTION_SENTENCE))
embeds_b = EMBED_CALLS["n"]
for q, r, ms in rows_b:
    print(f"  {ms:7.1f} ms  {r['cache'].upper():<5} key={cache.normalize_query(q)!r}")
print(f"  -> {len(WORKLOAD)} requests, {embeds_b} embedding+search+generation runs, {total_b:.0f} ms total")
print(f"  -> cache hits={cache.STATS['hits']} misses={cache.STATS['misses']} entries={cache.size()}")

mismatches = [a[0].strip() for a, b in zip(rows_a, rows_b)
              if a[1]["answer"] != b[1]["answer"] or a[1]["sources"] != b[1]["sources"]]
same = not mismatches
print("\n=== EVIDENCE ===")
print(f"  grounded-generation runs : {embeds_a} -> {embeds_b}   ({embeds_a - embeds_b} avoided)")
print(f"  total time               : {total_a:.0f} ms -> {total_b:.0f} ms   ({total_a / max(total_b, 1e-9):.1f}x faster)")
print(f"  every cached answer identical to the uncached one: {same}   {mismatches or ''}")
assert embeds_b == 4 and cache.STATS["hits"] == len(WORKLOAD) - 4, "unexpected cache counts"

print("\n=== INVALIDATION: the cache is cleared when the knowledge base changes ===")
from api import documents
q = "How much does gift wrapping cost?"
r1 = cache.cached_answer_query(q, COLLECTION_SENTENCE)
print(f"  before adding the document: fallback={r1['fallback']}  (cache entries: {cache.size()})")
info = documents.add_document("Gift wrapping policy", "Gift wrapping costs Rs 49 per item and is available on Beauty and Home orders. "
                              "Gift wrapped orders cannot be returned without the wrapping.")
print(f"  document added ({info['doc_id']}); cache entries now: {cache.size()}")
r2 = cache.cached_answer_query(q, COLLECTION_SENTENCE)
print(f"  same question again: cache={r2['cache']} fallback={r2['fallback']}  answer: {r2['answer'][:90]}")
assert r2["cache"] == "miss" and not r2["fallback"]
documents.delete_document(info["doc_id"])
print(f"  document deleted; cache entries now: {cache.size()}")