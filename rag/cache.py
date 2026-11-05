"""
Task 16 - in-memory response cache for the grounded-generation step (rag.grounded.answer_query).

Key   : (collection, k, normalized query) - lower-cased with whitespace collapsed. The embedding model
        (all-MiniLM-L6-v2) is uncased and ignores extra spaces, so these variants give the IDENTICAL vector and
        therefore the identical answer. Punctuation is kept because it does change the embedding.
Value : the full answer dict.  Max 256 entries (oldest evicted first).
The cache is CLEARED whenever the knowledge base changes (add / delete document), otherwise a cached
"I don't know" could outlive the document that now answers it.
"""
import re
import threading
from collections import OrderedDict

MAX_ENTRIES = 256
_CACHE: "OrderedDict[tuple, dict]" = OrderedDict()
_LOCK = threading.Lock()
STATS = {"hits": 0, "misses": 0}


def normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query.lower()).strip()


def clear() -> None:
    with _LOCK:
        _CACHE.clear()


def size() -> int:
    return len(_CACHE)


def reset_stats() -> None:
    STATS.update(hits=0, misses=0)


def cached_answer_query(query: str, collection_name: str | None = None, k: int | None = None, answer_fn=None) -> dict:
    """Same result as answer_query(); the second call for an equivalent query skips embedding + search + generation."""
    if answer_fn is None:
        from rag.config import COLLECTION_SENTENCE, TOP_K
        from rag.grounded import answer_query as answer_fn
        collection_name = collection_name or COLLECTION_SENTENCE
        k = k or TOP_K
    key = (collection_name, k, normalize_query(query))
    with _LOCK:
        if key in _CACHE:
            STATS["hits"] += 1
            _CACHE.move_to_end(key)
            return {**_CACHE[key], "cache": "hit"}
    result = answer_fn(query, collection_name, k)
    with _LOCK:
        STATS["misses"] += 1
        _CACHE[key] = result
        while len(_CACHE) > MAX_ENTRIES:
            _CACHE.popitem(last=False)
    return {**result, "cache": "miss"}