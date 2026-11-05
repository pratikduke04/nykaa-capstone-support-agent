"""
Task 15 - least autonomy: only the Lookup Agent may call check_order_status.

Two layers:
  1. DESIGN  - each agent is built with only the tools it needs (see crew/crew_builder.py).
  2. ENFORCEMENT - call_tool_as() is the single gateway: an agent can run a tool only if that tool is in ITS OWN
     tool list. Anything else is refused and written to the audit log. Nothing is run for a refused call.
"""
from crew.crew_builder import ALL_AGENTS, composer_agent, lookup_agent, retrieval_agent  # noqa: F401

SENSITIVE_TOOL = "check_order_status"
AUDIT: list[dict] = []


class ToolAccessDenied(PermissionError):
    pass


def tools_of(agent) -> list[str]:
    return [t.name for t in (agent.tools or [])]


def holders_of(tool_name: str) -> list[str]:
    return [a.role for a in ALL_AGENTS if tool_name in tools_of(a)]


def call_tool_as(agent, tool_name: str, **kwargs):
    tool = next((t for t in (agent.tools or []) if t.name == tool_name), None)
    AUDIT.append({"agent": agent.role, "tool": tool_name, "allowed": tool is not None})
    if tool is None:
        raise ToolAccessDenied(f"{agent.role!r} is not allowed to use {tool_name!r}; it holds {tools_of(agent) or 'no tools'}")
    return tool.run(**kwargs)