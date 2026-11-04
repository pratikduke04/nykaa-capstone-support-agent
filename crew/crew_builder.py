"""
Task 7 - the 3-agent CrewAI crew, run through crew.kickoff().

  Retrieval Agent   - equipped ONLY with rag_lookup
  Lookup Agent      - equipped ONLY with check_order_status   (least autonomy: nobody else has it)
  Response Composer - no tools; combines both outputs into one AnswerSchema draft

The Lookup task is added only when the question contains an order id (no id -> nothing to look up).
Everything runs on the MockLLM (zero API keys, zero network).
"""
import os

from crewai import Agent, Crew, Process, Task


from crew.compose import extract_tagged
from crew.mock_llm import CALL_LOG, MockLLM
from crew.mock_logic import ORDER_ID_RE, strip_identifiers
from crew.settings import no_trace_kwargs, structured_output_kwargs
from crew.tools import CheckOrderStatusTool, RagLookupTool, TOOL_CALLS


def get_llm():
    if os.environ.get("MOCK_LLM", "1") == "1":
        return MockLLM(model="mock-llm", provider="mock", context_window_size=128000)
    raise NotImplementedError("Only MOCK_LLM=1 is wired. A real LLM would be created here behind MOCK_LLM=0.")


_llm = get_llm()

retrieval_agent = Agent(
    role="Retrieval Agent",
    goal="Find the Nykaa policy text that answers the customer's question.",
    backstory="You search the Nykaa policy knowledge base and report exactly what it contains.",
    tools=[RagLookupTool()], llm=_llm, allow_delegation=False, verbose=False)

lookup_agent = Agent(
    role="Lookup Agent",
    goal="Look up the status of the customer's order.",
    backstory="You are the only agent allowed to query the order system.",
    tools=[CheckOrderStatusTool()], llm=_llm, allow_delegation=False, verbose=False)

composer_agent = Agent(
    role="Response Composer",
    goal="Write one clear, grounded answer for the customer.",
    backstory="You combine the policy text and the order data you are given. You never invent facts.",
    tools=[], llm=_llm, allow_delegation=False, verbose=False)

ALL_AGENTS = [retrieval_agent, lookup_agent, composer_agent]


def run_crew(effective_question: str) -> dict:
    """Run the crew once. Returns raw composer output, both tool outputs, and evidence counters."""
    tools_before, llm_before = len(TOOL_CALLS), len(CALL_LOG)
    q = effective_question

    t_retrieve = Task(
        description=f"Search the policy knowledge base for the customer's question. "
                    f"[[QUESTION]]{strip_identifiers(q) or q}[[/QUESTION]]",
        expected_output="The policy search result.", agent=retrieval_agent)
    tasks, context = [t_retrieve], [t_retrieve]

    t_lookup = None
    if ORDER_ID_RE.search(q):
        t_lookup = Task(
            description=f"Look up the order mentioned by the customer. [[QUESTION]]{q}[[/QUESTION]]",
            expected_output="The order status result.", agent=lookup_agent)
        tasks.append(t_lookup)
        context.append(t_lookup)

    t_compose = Task(
        description=f"Write the final answer for the customer using the context from the earlier tasks. "
                    f"[[QUESTION]]{q}[[/QUESTION]]",
        expected_output="A JSON object that matches the AnswerSchema.",
        agent=composer_agent, context=context, **structured_output_kwargs())
    tasks.append(t_compose)

    crew = Crew(agents=ALL_AGENTS, tasks=tasks, process=Process.sequential, verbose=False, **no_trace_kwargs())
    output = crew.kickoff()                                   # <-- the .kickoff() call

    def tagged(task, tag):
        raw = getattr(getattr(task, "output", None), "raw", None) if task else None
        return extract_tagged([{"role": "user", "content": raw}], tag) if raw else None

    return {
        "raw": output.raw,
        "retrieval": tagged(t_retrieve, "RETRIEVAL"),
        "lookup": tagged(t_lookup, "LOOKUP"),
        "tools_invoked": [c["tool"] for c in TOOL_CALLS[tools_before:]],
        "llm_calls": len(CALL_LOG) - llm_before,
        "structured_output": "CrewAI output_pydantic + code validation" if structured_output_kwargs()
                             else "code validation only",
    }