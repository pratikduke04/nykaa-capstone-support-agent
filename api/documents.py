"""
POST /add-document and DELETE /documents/{id} logic.

Safety rules for documents added at runtime (they feed retrieval, so they are an attack surface):
  * text that matches the prompt-injection patterns is rejected;
  * PII (phone, card last-4) is masked BEFORE the document is stored or embedded;
  * added documents live in kb_added/ - the original 12 policy documents can never be deleted here;
  * document ids are validated, so no path tricks are possible.
"""
import re

from crew.compose import split_sentences
from crew.guardrails import detect_injection, mask_pii
from rag.config import COLLECTION_FIXED, COLLECTION_SENTENCE, KB_ADDED_DIR, KB_DIR

DOC_ID_RE = re.compile(r"^\d{2,}_[a-z0-9_]+$")


class RejectedDocument(ValueError):
    pass


class DocumentNotFound(KeyError):
    pass


class ProtectedDocument(PermissionError):
    pass


def slugify(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")[:50] or "document"


def next_number(existing_ids) -> int:
    nums = [int(m.group(1)) for i in existing_ids if (m := re.match(r"(\d+)_", i))]
    return (max(nums) + 1) if nums else 1


def prepare_document(title: str, text: str, existing_ids) -> tuple[dict, dict]:
    """Validate + mask. Pure function (no embeddings), so it is unit-testable. Returns (doc, pii_counts)."""
    if detect_injection(f"{title} {text}"):
        raise RejectedDocument("Document rejected: it contains prompt-injection style instructions.")
    clean_title, pii_t = mask_pii(title.strip())
    clean_text, pii_x = mask_pii(" ".join(text.split()))
    if not split_sentences(clean_text):
        raise RejectedDocument("Document rejected: it has no readable sentences.")
    pii = {k: pii_t.get(k, 0) + pii_x.get(k, 0) for k in set(pii_t) | set(pii_x)}
    doc_id = f"{next_number(existing_ids):02d}_{slugify(clean_title)}"
    return {"doc_id": doc_id, "title": clean_title, "text": clean_text}, pii


def _refresh_caches():
    import crew.compose as compose
    import crew.tools as tools
    tools._DOCS = None               # parent-document map used by the RAG tool
    compose._KB_STATS = None         # keyword statistics used by the composer


def add_document(title: str, text: str) -> dict:
    from rag.chunking import chunk_fixed, chunk_sentences, load_documents
    from rag.grounded import shared_client
    from rag.index import index_chunks

    doc, pii = prepare_document(title, text, [d["doc_id"] for d in load_documents()])
    KB_ADDED_DIR.mkdir(exist_ok=True)
    (KB_ADDED_DIR / f"{doc['doc_id']}.md").write_text(f"# {doc['title']}\n{doc['text']}\n", encoding="utf-8")

    fixed, sent = chunk_fixed(doc), chunk_sentences(doc)
    client = shared_client()
    index_chunks(client, COLLECTION_FIXED, fixed)
    index_chunks(client, COLLECTION_SENTENCE, sent)
    _refresh_caches()
    return {"doc_id": doc["doc_id"], "title": doc["title"], "chunks_fixed": len(fixed),
            "chunks_sentence": len(sent), "pii_masked": pii}


def delete_document(doc_id: str) -> None:
    from rag.grounded import shared_client
    from rag.index import get_collection

    if not DOC_ID_RE.match(doc_id):
        raise DocumentNotFound(doc_id)
    if (KB_DIR / f"{doc_id}.md").exists():
        raise ProtectedDocument("The original knowledge-base documents cannot be deleted.")
    path = KB_ADDED_DIR / f"{doc_id}.md"
    if not path.exists():
        raise DocumentNotFound(doc_id)
    client = shared_client()
    for name in (COLLECTION_FIXED, COLLECTION_SENTENCE):
        get_collection(client, name).delete(where={"doc_id": doc_id})
    path.unlink()
    _refresh_caches()