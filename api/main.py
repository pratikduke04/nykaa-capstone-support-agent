"""
Tasks 11-12 - FastAPI service: 2+ HTTP endpoints, a WebSocket chat, JSON-Lines request logging.

Run:  uvicorn api.main:app --port 8000
"""
import threading
import time
import uuid

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from pydantic import ValidationError

import crew  # noqa: F401  (sets MOCK_LLM + telemetry/tracing env vars BEFORE crewai is imported)
from api import documents
from api.models import (AddDocumentRequest, AddDocumentResponse, AskRequest, AskResponse,
                        DeleteDocumentResponse, HealthResponse)
from api.request_log import log_request
from crew.pipeline import answer_question

app = FastAPI(title="Nykaa Support Agent", version="1.0")

# The crew keeps module-level call logs, so only one crew run may execute at a time.
CREW_LOCK = threading.Lock()


def _ask_sync(session_id: str, text: str, trace_id: str) -> dict:
    with CREW_LOCK:
        return answer_question(session_id, text, trace_id=trace_id)


def _log_fields(result: dict) -> dict:
    resp = result["response"]
    return {
        "request_text": result["masked_question"],       # masked text only - never the raw input
        "pii_masked": result["pii_hits"],
        "blocked": result["blocked"],
        "intent": resp["intent"],
        "grounded": resp["grounded"],
        "tools_invoked": result["tools_invoked"],
        "llm_calls": result["llm_calls"],
        "output_guardrail": result["output_guardrail"]["action"],
        "pipeline_ms": result["latency_ms"],
    }


@app.middleware("http")
async def trace_and_log(request: Request, call_next):
    """One JSON-Lines entry per HTTP request: trace id, timing, status, plus fields the endpoint adds."""
    trace_id = uuid.uuid4().hex[:12]
    request.state.trace_id = trace_id
    request.state.log = {}
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["X-Trace-Id"] = trace_id
        return response
    finally:
        log_request({
            "channel": "http", "event": "request", "trace_id": trace_id,
            "method": request.method, "path": request.url.path, "status_code": status,
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
            **request.state.log,
        })


@app.get("/health", response_model=HealthResponse)
def health():
    import os
    return HealthResponse(status="ok", mock_llm=os.environ.get("MOCK_LLM", "1"))


@app.post("/ask", response_model=AskResponse)
def ask(body: AskRequest, request: Request):
    result = _ask_sync(body.session_id, body.question, request.state.trace_id)
    request.state.log.update(session_id=body.session_id, **_log_fields(result))
    return AskResponse(
        trace_id=request.state.trace_id, session_id=body.session_id,
        masked_question=result["masked_question"], pii_masked=result["pii_hits"],
        blocked=result["blocked"], tools_invoked=result["tools_invoked"],
        output_guardrail=result["output_guardrail"], latency_ms=result["latency_ms"],
        answer=result["response"],
    )


@app.post("/add-document", response_model=AddDocumentResponse, status_code=201)
def add_document(body: AddDocumentRequest, request: Request):
    try:
        info = documents.add_document(body.title, body.text)
    except documents.RejectedDocument as exc:
        request.state.log.update(event="add_document_rejected", reason=str(exc))
        raise HTTPException(status_code=400, detail=str(exc))
    request.state.log.update(event="add_document", doc_id=info["doc_id"], pii_masked=info["pii_masked"])
    return AddDocumentResponse(trace_id=request.state.trace_id, **info)


@app.delete("/documents/{doc_id}", response_model=DeleteDocumentResponse)
def delete_document(doc_id: str, request: Request):
    try:
        documents.delete_document(doc_id)
    except documents.DocumentNotFound:
        raise HTTPException(status_code=404, detail="Document not found.")
    except documents.ProtectedDocument as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    request.state.log.update(event="delete_document", doc_id=doc_id)
    return DeleteDocumentResponse(trace_id=request.state.trace_id, doc_id=doc_id, deleted=True)


@app.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket, session_id: str = "ws-default"):
    """Multi-turn chat. A client leaving (cleanly or abruptly) ends only THIS connection."""
    await websocket.accept()
    conn = uuid.uuid4().hex[:12]
    log_request({"channel": "websocket", "event": "connect", "trace_id": conn, "session_id": session_id})
    turns = 0
    try:
        while True:
            text = await websocket.receive_text()
            trace_id = uuid.uuid4().hex[:12]
            start = time.perf_counter()
            try:
                req = AskRequest(question=text, session_id=session_id)
            except ValidationError:
                await websocket.send_json({"trace_id": trace_id, "error": "Message must be 1-2000 characters."})
                log_request({"channel": "websocket", "event": "message_rejected", "trace_id": trace_id,
                             "connection_id": conn, "session_id": session_id, "status_code": 422,
                             "latency_ms": round((time.perf_counter() - start) * 1000, 2)})
                continue
            result = await run_in_threadpool(_ask_sync, req.session_id, req.question, trace_id)
            turns += 1
            log_request({"channel": "websocket", "event": "message", "trace_id": trace_id,
                         "connection_id": conn, "session_id": session_id, "status_code": 200,
                         "latency_ms": round((time.perf_counter() - start) * 1000, 2),
                         **_log_fields(result)})
            await websocket.send_json({"trace_id": trace_id, "masked_question": result["masked_question"],
                                       "answer": result["response"]})
    except (WebSocketDisconnect, RuntimeError):
        pass  # client went away; fall through to the disconnect log, server keeps serving others
    finally:
        log_request({"channel": "websocket", "event": "disconnect", "trace_id": conn,
                     "session_id": session_id, "turns": turns})