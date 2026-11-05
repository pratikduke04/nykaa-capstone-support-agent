"""
Task 14 - Autogen review stage: Fact-Checker -> Final-Editor in a RoundRobinGroupChat (max_turns=2).

  * Fact-Checker  : plain-text report listing every sentence the evidence does not support.
  * Final-Editor  : output_content_type=Verdict, so its reply is a StructuredMessage[Verdict].
  * The team is told about that message type with custom_message_types=[StructuredMessage[Verdict]].

Run the demo:  python -m review.autogen_review
"""
import asyncio
import os

import crew  # noqa: F401  (MOCK_LLM + telemetry env vars)
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import StructuredMessage
from autogen_agentchat.teams import RoundRobinGroupChat

from rag.chunking import load_documents
from review.mock_client import MockChatClient
from review.review_logic import build_task
from review.schemas import Verdict

REVIEWER_PROMPT = ("[[ROLE:REVIEWER]] You are the Fact-Checker for a Nykaa support assistant. Compare every sentence "
                   "of the draft answer with the evidence (knowledge-base text and the order record). List each "
                   "sentence the evidence does not support. Never add new facts.")
EDITOR_PROMPT = ("[[ROLE:EDITOR]] You are the Final-Editor. Read the draft and the Fact-Checker's report. If nothing "
                 "is unsupported, approve the draft unchanged. Otherwise remove the unsupported claims and explain. "
                 "Reply with a Verdict: approved, final_answer, reason.")


def build_team(client) -> RoundRobinGroupChat:
    reviewer = AssistantAgent("fact_checker", model_client=client, system_message=REVIEWER_PROMPT,
                              description="Checks each claim against the evidence.")
    editor = AssistantAgent("final_editor", model_client=client, system_message=EDITOR_PROMPT,
                            output_content_type=Verdict,
                            description="Approves or corrects the answer and returns a structured Verdict.")
    return RoundRobinGroupChat([reviewer, editor], max_turns=2,
                               custom_message_types=[StructuredMessage[Verdict]])


async def review_answer(question: str, draft: str, docs: list[tuple[str, str]], order: dict | None = None,
                        client: MockChatClient | None = None) -> dict:
    client = client or MockChatClient()
    team = build_team(client)
    result = await team.run(task=build_task(question, draft, docs, order))
    verdict = next(m.content for m in reversed(result.messages) if isinstance(m, StructuredMessage))
    return {"verdict": verdict, "messages": result.messages, "model_calls": list(client.calls),
            "tokens": client.total_usage().prompt_tokens + client.total_usage().completion_tokens}


def evidence_for(doc_ids: list[str]) -> list[tuple[str, str]]:
    by_id = {d["doc_id"]: d["text"] for d in load_documents()}
    return [(i, by_id[i]) for i in doc_ids if i in by_id]


def review_pipeline_result(question: str, result: dict, client=None) -> dict:
    """Review a pipeline result (the dict returned by crew.pipeline.answer_question)."""
    r = result["response"]
    return asyncio.run(review_answer(question, r["answer"], evidence_for(r["sources"]), r["order"], client))


def _print(title: str, out: dict) -> None:
    print(f"\n=== {title} ===")
    for m in out["messages"]:
        body = m.content.model_dump() if isinstance(m, StructuredMessage) else m.content
        shown = body if isinstance(body, dict) else str(body)
        print(f"[{type(m).__name__} from {m.source}]")
        print(f"   {shown if isinstance(shown, dict) else shown[:420].replace(chr(10), chr(10) + '   ')}")
    v = out["verdict"]
    print(f"-> messages in the chat: {len(out['messages'])} (task + 2 agent turns), model calls: {out['model_calls']}")
    print(f"-> Verdict: approved={v.approved}")
    print(f"   reason: {v.reason}")


def main() -> None:
    from crew.pipeline import answer_question
    assert os.environ.get("MOCK_LLM") == "1"

    q = "What is the return window for apparel?"
    res = answer_question("review-a", q)
    draft = res["response"]["answer"]
    docs = evidence_for(res["response"]["sources"])      # the documents the answer cites

    out_a = asyncio.run(review_answer(q, draft, docs))
    _print("CASE A - grounded draft: expect APPROVE, answer unchanged", out_a)
    assert out_a["verdict"].approved and out_a["verdict"].final_answer == draft
    print("   check: final_answer == original draft  ->", out_a["verdict"].final_answer == draft)

    injected = "Refunds are guaranteed within 24 hours for every order."
    out_b = asyncio.run(review_answer(q, draft + " " + injected, docs))
    _print("CASE B - draft with an INJECTED ungrounded claim: expect REVISE", out_b)
    v = out_b["verdict"]
    assert not v.approved and injected not in v.final_answer
    print("   check: injected claim removed from final_answer ->", injected not in v.final_answer)
    print("   check: final_answer == original grounded draft  ->", v.final_answer == draft)

    q2 = "Where is my order NYK-1005?"
    res2 = answer_question("review-c", q2)
    out_c = review_pipeline_result(q2, res2)
    _print("CASE C - order-status answer checked against the order record: expect APPROVE", out_c)
    assert out_c["verdict"].approved

    print(f"\nTokens used by the review stage (word-count estimate): A={out_a['tokens']}  B={out_b['tokens']}  C={out_c['tokens']}")


if __name__ == "__main__":
    main()