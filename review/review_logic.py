"""
Task 14 - the review rules used by the mock model (pure Python, no autogen import, unit-testable).

Reviewer  : checks every sentence of the draft against the evidence (retrieved KB text + the order record).
Editor    : approves the draft unchanged, or removes the unsupported sentences and says why.

A policy sentence is supported if it appears in the evidence text. An order sentence is supported if its
numbers and status match the order record and it uses only order-record vocabulary. This is a heuristic
stand-in for a real LLM reviewer, which could also judge paraphrases.
"""
import json
import re

PREFIX = "According to our policy:"
STATUSES = ["placed", "shipped", "delivered", "returned", "refunded"]
ORDER_PHRASES = {"currently", "order", "value", "system", "recommends", "escalating", "escalation", "score",
                 "support", "team", "delayed", "shipment", "status", "your", "this", "that", "with", "from"}
FALLBACK = ("I don't have enough information in the Nykaa policy knowledge base "
            "to answer that. Please contact customer support.")


def norm(text: str) -> str:
    t = re.sub(r"(?<=\d),(?=\d)", "", str(text).lower())
    return re.sub(r"\s+", " ", t).strip()


def split_draft(draft: str) -> tuple[str, list[str]]:
    """Return (prefix, sentences). The 'According to our policy:' lead-in is not a claim."""
    prefix = PREFIX if draft.startswith(PREFIX) else ""
    text = draft[len(prefix):].strip()
    return prefix, [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def order_line(order: dict | None) -> str:
    if not order:
        return ""
    return (f"ORDER {order['record_id']} | status {order['status']} | order value {order['order_value_inr']} | "
            f"escalation score {order['escalation_score']:.2f} | escalate {'yes' if order['escalate_recommended'] else 'no'}")


def build_task(question: str, draft: str, docs: list[tuple[str, str]], order: dict | None) -> str:
    evidence = "\n".join(f"[{doc_id}] {text}" for doc_id, text in docs)
    return (f"Review this draft customer-support answer.\n"
            f"[[QUESTION]]{question}[[/QUESTION]]\n[[DRAFT]]{draft}[[/DRAFT]]\n"
            f"[[EVIDENCE]]{evidence}[[/EVIDENCE]]\n[[ORDER]]{order_line(order)}[[/ORDER]]")


def tag(text: str, name: str) -> str:
    hit = re.search(rf"\[\[{name}\]\](.*?)\[\[/{name}\]\]", text, re.DOTALL)
    return hit.group(1).strip() if hit else ""


def _order_supported(sentence: str, line: str) -> bool:
    if not line:
        return False
    s, o = norm(sentence), norm(line)
    nums = re.findall(r"\d+(?:\.\d+)?", s)
    if any(n not in re.findall(r"\d+(?:\.\d+)?", o) for n in nums):
        return False
    for st in STATUSES:
        if st in s and f"status {st}" not in o:
            return False
    if "escalat" in s and "escalate yes" not in o and "not" not in s:
        return False
    words = [w for w in re.findall(r"[a-z]+", s) if len(w) > 3]
    vocab = set(re.findall(r"[a-z]+", o)) | ORDER_PHRASES
    return not words or sum(w in vocab for w in words) / len(words) >= 0.8


def is_order_sentence(sentence: str) -> bool:
    s = norm(sentence)
    return "nyk-" in s or "escalat" in s


def unsupported_sentences(draft: str, evidence: str, line: str) -> tuple[int, list[str]]:
    _, sentences = split_draft(draft)
    ev = norm(evidence)
    bad = []
    for s in sentences:
        if is_order_sentence(s):
            ok = _order_supported(s, line)
        else:
            ok = norm(s) in ev
        if not ok:
            bad.append(s)
    return len(sentences), bad


def reviewer_report(task_text: str) -> str:
    draft, evidence, line = tag(task_text, "DRAFT"), tag(task_text, "EVIDENCE"), tag(task_text, "ORDER")
    n, bad = unsupported_sentences(draft, evidence, line)
    lines = [f"FACT-CHECK REPORT: checked {n} sentence(s) against the retrieved evidence."]
    lines += [f"UNSUPPORTED: {s}" for s in bad] or ["All sentences are supported by the evidence."]
    lines.append(f"[[UNGROUNDED]]{json.dumps(bad)}[[/UNGROUNDED]]")
    return "\n".join(lines)


def editor_verdict(all_text: str) -> dict:
    draft = tag(all_text, "DRAFT")
    bad = json.loads(tag(all_text, "UNGROUNDED") or "[]")
    if not bad:
        return {"approved": True, "final_answer": draft,
                "reason": "Every sentence is supported by the retrieved evidence; the draft is unchanged."}
    prefix, sentences = split_draft(draft)
    kept = [s for s in sentences if s not in bad]
    final = (f"{prefix} {' '.join(kept)}".strip()) if kept else FALLBACK
    return {"approved": False, "final_answer": final,
            "reason": f"Removed {len(bad)} unsupported claim(s): " + " | ".join(bad)}