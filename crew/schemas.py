"""
Task 9 - the Pydantic schema EVERY crew response must conform to.
No crewai import here, so it can be tested anywhere.
"""
from typing import Literal, Optional

from pydantic import BaseModel, Field

FALLBACK_ANSWER = ("I don't have enough information in the Nykaa policy knowledge base "
                   "to answer that. Please contact customer support.")


class OrderInfo(BaseModel):
    record_id: str
    status: str
    order_value_inr: int = Field(ge=0)
    escalation_score: float = Field(ge=0.0, le=1.0)
    escalate_recommended: bool


class AnswerSchema(BaseModel):
    """The structured response returned for every customer question."""
    intent: Literal["policy", "order", "policy_and_order", "out_of_scope", "blocked"]
    answer: str = Field(min_length=1)
    grounded: bool                                   # supported by retrieved context or tool data?
    top_similarity: float = Field(ge=-1.0, le=1.0)   # best chunk cosine similarity (0 if no retrieval)
    sources: list[str] = Field(default_factory=list) # parent document ids that support the answer
    order: Optional[OrderInfo] = None
    escalate: bool = False