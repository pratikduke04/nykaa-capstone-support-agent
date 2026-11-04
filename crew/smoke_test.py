"""
Smoke test: two single-agent crews run through crew.kickoff() with the MockLLM, proving that
  (1) the RAG tool is invoked for a policy question, and
  (2) check_order_status is invoked for an order question.

Run from the project root:   python -m crew.smoke_test
"""
from crewai import Agent, Crew, Process, Task

from crew.mock_llm import CALL_LOG, MockLLM
from crew.tools import CheckOrderStatusTool, RagLookupTool, TOOL_CALLS
from crew.settings import no_trace_kwargs

llm = MockLLM(model="mock-llm", provider="mock", context_window_size=128000)


def run_single(agent: Agent, description: str, question: str) -> str:
    task = Task(description=f"{description} [[QUESTION]]{question}[[/QUESTION]]",
                expected_output="The raw tool result, wrapped in tags.", agent=agent)
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False,
                **no_trace_kwargs())
    return crew.kickoff().raw


retrieval_agent = Agent(role="Retrieval Agent", goal="Find the relevant Nykaa policy text.",
                        backstory="You search the policy knowledge base.", tools=[RagLookupTool()],
                        llm=llm, allow_delegation=False, verbose=False)
lookup_agent = Agent(role="Lookup Agent", goal="Look up the status of one order.",
                     backstory="You check orders in the order system.", tools=[CheckOrderStatusTool()],
                     llm=llm, allow_delegation=False, verbose=False)

if __name__ == "__main__":
    q1 = "How long does a refund take for a cash on delivery order?"
    q2 = "Where is my order NYK-1045?"

    print("=" * 80, f"\nQUERY 1 (policy): {q1}\n" + "=" * 80)
    out1 = run_single(retrieval_agent, "Search the policy knowledge base.", q1)
    print("crew.kickoff() result:\n", out1[:700], "...\n")

    print("=" * 80, f"\nQUERY 2 (order): {q2}\n" + "=" * 80)
    out2 = run_single(lookup_agent, "Look up the order.", q2)
    print("crew.kickoff() result:\n", out2, "\n")

    print("=" * 80, "\nEVIDENCE - tools actually executed (TOOL_CALLS):")
    for c in TOOL_CALLS:
        print("  ", c)
    print("\nEVIDENCE - every LLM call the crew made (CALL_LOG):")
    for c in CALL_LOG:
        print("  ", c)