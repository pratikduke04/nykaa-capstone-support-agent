"""
Task 15 - Low / Medium / High risk classification.

Part 1 - a RISK REGISTER for what the system can do (capabilities). Each row is scored with the same rubric:
    data_sensitivity  0 = public policy text, 1 = derived from customer data, 2 = raw customer data
    writes            1 if it changes stored state
    blast_radius      0 = none, 1 = one customer, 2 = every customer
    irreversible      1 if it cannot be undone
    unauthenticated   1 if anyone who can reach it can use it
  total 0-1 = Low, 2-3 = Medium, 4+ = High.

Part 2 - classify_request(): the same idea applied to one incoming message.
"""
import re

REGISTER = [
    # name, what it does, (sensitivity, writes, blast, irreversible, unauth), mitigation
    ("rag_lookup", "Read policy text from the knowledge base", (0, 0, 0, 0, 0),
     "Read-only on public policy text."),
    ("request_logging", "Write one JSON line per request", (0, 1, 0, 0, 0),
     "Question text is PII-masked twice before it is written."),
    ("check_order_status", "Read one customer's order record", (2, 0, 0, 0, 0),
     "Read-only; only the Lookup Agent holds it; order ids are validated."),
    ("compose_answer", "Write the customer-facing reply", (1, 0, 1, 0, 0),
     "Output guardrail + Autogen fact-check review before sending."),
    ("escalation_recommendation", "Flag an order for human escalation (advice only)", (1, 0, 1, 0, 0),
     "Advisory flag with a documented score; a human takes the action."),
    ("add_document (API)", "Add a document that future answers will quote", (0, 1, 2, 0, 1),
     "Injection scan + PII masking; deletable. Needs authentication before production."),
    ("delete_document (API)", "Remove a knowledge-base document", (0, 1, 2, 1, 1),
     "Original 12 documents are protected. Needs authentication before production."),
    ("issue_refund (NOT built)", "Move money back to a customer", (2, 1, 1, 1, 0),
     "Deliberately out of scope: the agent is read-only. Would need human approval."),
]

LEVELS = [(4, "High"), (2, "Medium"), (0, "Low")]


def level_for(score: int) -> str:
    return next(name for threshold, name in LEVELS if score >= threshold)


def register() -> list[dict]:
    rows = []
    for name, what, (sens, wr, blast, irrev, unauth), mitigation in REGISTER:
        score = sens + wr + blast + irrev + unauth
        rows.append({"capability": name, "what": what, "score": score, "level": level_for(score),
                     "factors": f"sensitivity={sens} writes={wr} blast={blast} irreversible={irrev} unauth={unauth}",
                     "mitigation": mitigation})
    return rows


HIGH_WORDS = re.compile(r"\b(lawyer|legal|court|consumer forum|fraud|chargeback|police|sue)\b", re.I)
ORDER_ID = re.compile(r"\bNYK-\d+\b", re.I)
PERSONAL = re.compile(r"\b(my order|my refund|my payment|my account|my parcel)\b|\[PHONE\]|\[CARD", re.I)


def classify_request(masked_question: str, injection_detected: bool = False) -> dict:
    if injection_detected:
        return {"level": "High", "reason": "prompt-injection attempt: someone is trying to change the agent's behaviour"}
    if HIGH_WORDS.search(masked_question):
        return {"level": "High", "reason": "legal / fraud / dispute wording: needs a human (Level 3 in the escalation matrix)"}
    if ORDER_ID.search(masked_question) or PERSONAL.search(masked_question):
        return {"level": "Medium", "reason": "touches one customer's order or personal data (read-only lookup)"}
    return {"level": "Low", "reason": "general policy question answered from public knowledge-base text"}