"""Task 11 - Pydantic request/response models for the HTTP API."""
from pydantic import BaseModel, Field

from crew.schemas import AnswerSchema

SESSION_PATTERN = r"^[A-Za-z0-9_.-]+$"


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="The customer's question.")
    session_id: str = Field(default="default", min_length=1, max_length=64, pattern=SESSION_PATTERN,
                            description="Conversation id; the same id keeps multi-turn memory.")


class AskResponse(BaseModel):
    trace_id: str
    session_id: str
    masked_question: str = Field(description="The question as the agent saw it (PII masked).")
    pii_masked: dict[str, int]
    blocked: bool
    tools_invoked: list[str]
    output_guardrail: dict
    latency_ms: float
    answer: AnswerSchema


class AddDocumentRequest(BaseModel):
    title: str = Field(..., min_length=3, max_length=120)
    text: str = Field(..., min_length=40, max_length=3000)


class AddDocumentResponse(BaseModel):
    trace_id: str
    doc_id: str
    title: str
    chunks_fixed: int
    chunks_sentence: int
    pii_masked: dict[str, int]


class DeleteDocumentResponse(BaseModel):
    trace_id: str
    doc_id: str
    deleted: bool


class HealthResponse(BaseModel):
    status: str
    mock_llm: str