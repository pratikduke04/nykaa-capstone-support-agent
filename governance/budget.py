"""
Task 15 - per-request token / cost budget cap.

Every request gets an estimated cost BEFORE any agent runs. If the estimate is over the cap the request is
rejected right away: zero LLM calls, zero tool calls. Tokens are estimated as characters / 4 (the same
rule of thumb CrewAI used in its own log line), plus a fixed allowance for the agents' prompt scaffolding
and the answer. MOCK_LLM costs nothing, so the price is a notional planning rate.
"""
import math
from dataclasses import asdict, dataclass

CHARS_PER_TOKEN = 4
RESERVED_TOKENS = 150                 # prompt scaffolding + the answer the crew will write
MAX_TOKENS_PER_REQUEST = 400
PRICE_PER_1K_TOKENS_INR = 0.40        # notional rate for planning
MAX_COST_PER_REQUEST_INR = 0.20


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


@dataclass
class BudgetDecision:
    allowed: bool
    est_question_tokens: int
    est_total_tokens: int
    est_cost_inr: float
    reason: str

    def as_dict(self) -> dict:
        return asdict(self)


def check_budget(masked_question: str) -> BudgetDecision:
    q_tokens = estimate_tokens(masked_question)
    total = q_tokens + RESERVED_TOKENS
    cost = round(total / 1000 * PRICE_PER_1K_TOKENS_INR, 4)
    if total > MAX_TOKENS_PER_REQUEST:
        return BudgetDecision(False, q_tokens, total, cost,
                              f"estimated {total} tokens exceeds the cap of {MAX_TOKENS_PER_REQUEST} per request")
    if cost > MAX_COST_PER_REQUEST_INR:
        return BudgetDecision(False, q_tokens, total, cost,
                              f"estimated cost Rs {cost} exceeds the cap of Rs {MAX_COST_PER_REQUEST_INR} per request")
    return BudgetDecision(True, q_tokens, total, cost, "within budget")