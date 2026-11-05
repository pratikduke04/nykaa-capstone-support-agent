"""Task 15 - one entry point that applies the governance checks in front of the normal pipeline."""
import crew  # noqa: F401
from crew.guardrails import check_input
from crew.pipeline import answer_question
from governance.budget import check_budget, estimate_tokens
from governance.risk import classify_request


def governed_answer(session_id: str, raw_text: str) -> dict:
    check = check_input(raw_text)                                   # masks PII first
    decision = check_budget(check.masked_text)
    risk = classify_request(check.masked_text, bool(check.injection_hits))
    if not decision.allowed:                                        # rejected BEFORE any agent runs
        return {"rejected": True, "reason": decision.reason, "budget": decision.as_dict(), "risk": risk,
                "llm_calls": 0, "tools_invoked": [], "response": None}
    result = answer_question(session_id, raw_text)
    answer = result["response"]["answer"]
    used = estimate_tokens(check.masked_text) + estimate_tokens(answer)
    return {**result, "rejected": False, "budget": decision.as_dict(), "risk": risk, "answer_tokens_est": used}