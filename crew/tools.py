"""
The two CrewAI tools:
  rag_lookup         - policy search over the kb_fixed collection (our recommended chunking strategy),
                       expanded from matching chunks to their PARENT documents so answers are clean text.
  check_order_status - wraps crew/order_status.py (Task 6).

Both tools append to TOOL_CALLS when they actually run - that list is the proof, in the transcripts,
that each tool was really invoked through crew.kickoff().
"""
import json
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from crew.order_status import check_order_status
from rag.chunking import load_documents
from rag.config import COLLECTION_FIXED, RELATIVE_MARGIN, TOP_K
from rag.grounded import get_threshold, retrieve

TOOL_CALLS: list[dict] = []
_DOCS: dict | None = None


def _docs_by_id() -> dict:
    global _DOCS
    if _DOCS is None:
        _DOCS = {d["doc_id"]: d for d in load_documents()}
    return _DOCS


def retrieve_parent_documents(query: str, collection: str = COLLECTION_FIXED, k: int = TOP_K) -> dict:
    """Top-k chunks -> keep those near the best score -> map to parent documents (dedup)."""
    chunks = retrieve(query, collection, k)
    threshold = get_threshold(collection)
    top = chunks[0]["similarity"] if chunks else 0.0
    grounded = top >= threshold
    docs, seen = [], set()
    if grounded:
        floor = max(threshold, top - RELATIVE_MARGIN)
        for c in chunks:
            if c["similarity"] < floor or c["doc_id"] in seen:
                continue
            seen.add(c["doc_id"])
            d = _docs_by_id()[c["doc_id"]]
            docs.append({"doc_id": c["doc_id"], "title": d["title"],
                         "similarity": round(c["similarity"], 3), "text": d["text"]})
    return {"query": query, "top_similarity": round(top, 3), "threshold": threshold,
            "grounded": grounded, "documents": docs}


class RagInput(BaseModel):
    query: str = Field(..., description="The customer's policy question, in plain words.")


class OrderInput(BaseModel):
    record_id: str = Field(..., description="Order id such as NYK-1045.")


class RagLookupTool(BaseTool):
    name: str = "rag_lookup"
    description: str = ("Search the Nykaa policy knowledge base (returns policies, refunds, delivery, "
                        "warranty, exchanges...). Input: the customer's question.")
    args_schema: Type[BaseModel] = RagInput

    def _run(self, query: str) -> str:
        TOOL_CALLS.append({"tool": self.name, "args": {"query": query}})
        return json.dumps(retrieve_parent_documents(query))


class CheckOrderStatusTool(BaseTool):
    name: str = "check_order_status"
    description: str = ("Look up one order by id (e.g. NYK-1045). Returns status, order value in INR, "
                        "escalation score and whether escalation is recommended.")
    args_schema: Type[BaseModel] = OrderInput

    def _run(self, record_id: str) -> str:
        TOOL_CALLS.append({"tool": self.name, "args": {"record_id": record_id}})
        return json.dumps(check_order_status(record_id))