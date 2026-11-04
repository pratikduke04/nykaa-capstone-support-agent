"""
The whole request pipeline for one customer message (Parts 2-4 plug into this):

  raw text -> INPUT GUARDRAILS (mask PII, block injection) -> session memory -> CREW (3 agents)
           -> validate against AnswerSchema -> OUTPUT GUARDRAIL (groundedness) -> result dict

Only the MASKED question travels past the first step.
"""
import time
import uuid

from pydantic import ValidationError

from crew.guardrails import blocked_response, check_input, check_output
from crew.memory import build_chain, resolve_references
from crew.schemas import AnswerSchema


def parse_answer(raw: str) -> AnswerSchema:
    """Task 9: every crew response is validated against AnswerSchema in code."""
    try:
        return AnswerSchema.model_validate_json(raw)
    except (ValidationError, ValueError):
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            return AnswerSchema.model_validate_json(raw[start:end + 1])
        raise


def run_turn(masked_question: str, history: list, runner) -> dict:
    effective, carried_id = resolve_references(masked_question, history)
    run = runner(effective)
    draft = parse_answer(run["raw"])                                   # Task 9
    final, verdict = check_output(draft, run["retrieval"])             # Task 10 (output side)
    return {"effective_question": effective, "carried_order_id": carried_id,
            "tools_invoked": run["tools_invoked"], "llm_calls": run["llm_calls"],
            "schema_valid": True, "output_guardrail": verdict, "response": final.model_dump()}


def _default_runner():
    from crew.crew_builder import run_crew        # imported lazily: only needs crewai when a crew really runs
    return run_crew


def answer_question(session_id: str, raw_text: str, runner=None, trace_id: str | None = None) -> dict:
    t0 = time.perf_counter()
    trace_id = trace_id or uuid.uuid4().hex[:12]
    check = check_input(raw_text)                                       # Task 10 (input side)
    base = {"trace_id": trace_id, "session_id": session_id, "masked_question": check.masked_text,
            "pii_hits": check.pii_hits, "injection_patterns_matched": len(check.injection_hits)}

    if check.blocked:
        return {**base, "blocked": True, "effective_question": None, "carried_order_id": None,
                "tools_invoked": [], "llm_calls": 0, "schema_valid": True,
                "output_guardrail": {"action": "not_reached", "reasons": []},
                "response": blocked_response(check).model_dump(),
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}

    runner = runner or _default_runner()
    chain = build_chain(lambda inputs: _inner(inputs, runner))
    out = chain.invoke({"input": check.masked_text}, config={"configurable": {"session_id": session_id}})
    return {**base, "blocked": False, **out["result"],
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}


def _inner(inputs: dict, runner) -> dict:
    result = run_turn(inputs["input"], inputs["history"], runner)
    return {"output": result["response"]["answer"], "result": result}