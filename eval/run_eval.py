"""
Task 13 - run the 15-query evaluation through the full pipeline and score it with the judge.

    python -m eval.run_eval 2>/dev/null > transcripts/task13_eval.txt
"""
import json
import os
from pathlib import Path

import crew  # noqa: F401  (MOCK_LLM + telemetry env vars)
from crew.order_status import compute_escalation_score, ORDERS_BY_ID, ESCALATION_THRESHOLD
from crew.pipeline import answer_question
from eval.judge import build_judge, judge_case
from eval.rubric import CRITERIA
from eval.test_set import CASES
from rag.chunking import load_documents

OUT = Path(__file__).resolve().parent.parent / "logs" / "eval_results.json"


def order_truth(order_id: str) -> dict:
    o = ORDERS_BY_ID[order_id]
    score = compute_escalation_score(o)
    return {"record_id": order_id, "status": o["status"], "order_value_inr": o["order_value_inr"],
            "escalation_score": score, "escalate_recommended": score >= ESCALATION_THRESHOLD}


def sanity_check(judge, docs):
    """Prove the judge can FAIL an answer: feed it three deliberately bad answers."""
    print("\n" + "=" * 78)
    print("JUDGE SANITY CHECK - deliberately bad answers must score low")
    q01, q13, q15 = CASES[0], CASES[12], CASES[14]
    bad = [
        ("Q01 hallucinated number, wrong source", q01, "Apparel can be returned within 30 days, no questions asked.",
         "policy", ["03_delivery_slas"], q01["query"]),
        ("Q13 answers an out-of-scope question", q13, "The capital of France is Paris.",
         "policy", ["01_return_window_by_category"], q13["query"]),
        ("Q15 repeats the customer's phone number", q15, "Order NYK-1005 is Delivered. We will call you on 9876543210.",
         "order", [], q15["query"]),
    ]
    for label, case, answer, intent, sources, mq in bad:
        payload = {"case": case, "answer": answer, "intent": intent, "sources": sources, "order": None,
                   "masked_question": mq, "source_texts": [docs[s] for s in sources if s in docs],
                   "order_truth": order_truth(case["order_id"]) if case.get("order_id") else None}
        v = judge_case(judge, payload)
        print(f"  {label}: accuracy={v.accuracy} grounding={v.grounding} completeness={v.completeness} safety={v.safety}")


def main():
    assert os.environ.get("MOCK_LLM") == "1", "run with MOCK_LLM=1 (the default)"
    docs = {d["doc_id"]: d["text"] for d in load_documents()}
    topics = {c["doc"] for c in CASES if c["doc"]}
    assert len(CASES) >= 15 and len(topics) == 12, "need >=15 queries covering all 12 KB topics"
    assert sum(c["type"] in ("refuse", "blocked") for c in CASES) >= 2, "need >=2 out-of-scope/edge cases"

    judge = build_judge()
    rows = []
    print(f"MOCK_LLM={os.environ['MOCK_LLM']}  queries={len(CASES)}  judge={type(judge).__name__}\n")
    for case in CASES:
        res = answer_question(f"eval-{case['id']}", case["query"])
        r = res["response"]
        payload = {"case": case, "answer": r["answer"], "intent": r["intent"], "sources": r["sources"],
                   "order": r["order"], "masked_question": res["masked_question"],
                   "source_texts": [docs[s] for s in r["sources"] if s in docs],
                   "order_truth": order_truth(case["order_id"]) if case.get("order_id") else None}
        v = judge_case(judge, payload)
        rows.append({"id": case["id"], "topic": case["topic"], "query": res["masked_question"], "intent": r["intent"],
                     "sources": r["sources"], "answer": r["answer"],
                     **{k: getattr(v, k) for k in CRITERIA}, "reasons": v.reasons})
        print(f"{case['id']}  {case['topic']}")
        print(f"   query : {res['masked_question']}")
        print(f"   answer: {r['answer'][:230]}")
        print(f"   intent={r['intent']} sources={r['sources']}")
        print(f"   scores: accuracy={v.accuracy} grounding={v.grounding} completeness={v.completeness} safety={v.safety}")
        for k in CRITERIA:
            print(f"     - {k}: {v.reasons.get(k, '')}")
        print()

    print("=" * 78)
    print(f"{'ID':<5}{'Topic':<30}{'Acc':>5}{'Grnd':>6}{'Comp':>6}{'Safe':>6}")
    for r in rows:
        print(f"{r['id']:<5}{r['topic'][:28]:<30}{r['accuracy']:>5}{r['grounding']:>6}{r['completeness']:>6}{r['safety']:>6}")
    avgs = {k: round(sum(r[k] for r in rows) / len(rows), 2) for k in CRITERIA}
    print("-" * 78)
    print(f"{'AVERAGE':<35}{avgs['accuracy']:>5}{avgs['grounding']:>6}{avgs['completeness']:>6}{avgs['safety']:>6}")
    weak = [(r["id"], k) for r in rows for k in CRITERIA if r[k] <= 3]
    print(f"\nScores of 3 or below: {weak or 'none'}")
    sanity_check(judge, docs)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"averages": avgs, "rows": rows}, indent=2), encoding="utf-8")
    print(f"Full results saved to {OUT.relative_to(OUT.parent.parent)}")


if __name__ == "__main__":
    main()