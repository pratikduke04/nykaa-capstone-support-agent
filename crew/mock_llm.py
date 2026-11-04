"""
MockLLM - a deterministic, keyless LLM for CrewAI (MOCK_LLM mode).

Extends crewai.llms.base_llm.BaseLLM (CrewAI's documented extension point). In CrewAI 1.15.x the only
abstract method is call(). The decision logic lives in crew/mock_logic.py (pure Python, unit-tested).

supports_function_calling() returns False on purpose:
  * CrewAI's agent code then uses the text-based ReAct loop, whose exact text we control;
  * crewai/utilities/converter.py calls llm.supports_function_calling() WITHOUT a hasattr() guard
    (needed for Pydantic structured output, Task 9), so the method must exist.

Set MOCK_TRACE=1 to append one JSON line per call to logs/mock_trace.jsonl (debugging aid).
"""
import json
import os
import time
from pathlib import Path
from typing import Any

from crewai.llms.base_llm import BaseLLM

from crew import mock_logic as ml

CALL_LOG: list[dict] = []          # one entry per LLM call - used as evidence in the transcripts
TRACE_FILE = Path(__file__).resolve().parent.parent / "logs" / "mock_trace.jsonl"


class MockLLM(BaseLLM):
    """Deterministic stand-in for a real model. No network, no API key."""

    def supports_function_calling(self) -> bool:
        return False

    def call(
        self,
        messages,
        tools=None,
        callbacks=None,
        available_functions=None,
        from_task=None,
        from_agent=None,
        response_model=None,
    ) -> Any:
        msgs = ml.normalize(messages)
        agent_tools = list(getattr(from_agent, "tools", None) or [])
        role = getattr(from_agent, "role", "unknown-agent")

        result = ml.decide(msgs, agent_tools)
        text = result["text"]

        if result["decision"] == "no_tools":              # e.g. the Response Composer
            try:
                from crew.compose import compose_final_answer     # written in the next step
                text = ml.format_final(compose_final_answer(msgs, result["question"]))
            except ImportError:
                text = ml.format_final("(composer not built yet) " + result["question"])

        entry = {"agent": role, "decision": result["decision"], "n_messages": len(msgs)}
        CALL_LOG.append(entry)
        if os.environ.get("MOCK_TRACE") == "1":
            TRACE_FILE.parent.mkdir(exist_ok=True)
            with TRACE_FILE.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    **entry, "ts": time.time(),
                    "roles": [m["role"] for m in msgs],
                    "last_message_head": msgs[-1]["content"][:300] if msgs else "",
                    "output_head": (text or "")[:300],
                }) + "\n")
        return text