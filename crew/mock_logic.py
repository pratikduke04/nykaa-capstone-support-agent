"""
Pure-Python decision logic for the MockLLM (no crewai import, so it can be unit-tested anywhere).

CrewAI's ReAct loop talks to the LLM like this:
  turn 1  -> model must answer   "Thought: ..\nAction: <tool>\nAction Input: {json}"
  (CrewAI runs the tool and sends the result back as   "Observation: <result>")
  turn 2  -> model must answer   "Thought: I now know the final answer\nFinal Answer: <text>"

Two silent traps from the brief are handled here:
  1. CrewAI's own prompt template contains the example text "Observation: the result of the action".
     So we NEVER search the whole conversation for "Observation:". We only (a) look at the model's OWN
     earlier turns (role == assistant) to learn whether a tool already ran, and (b) read the result from
     the LAST message only.
  2. A tool is never classified by its NAME (a tool called `rag_lookup` contains "lookup" but is not the
     order lookup). We classify by the tool's declared argument schema: a `record_id` field => order lookup.
"""
import json
import re

ORDER_ID_RE = re.compile(r"\bNYK-\d{4}\b", re.IGNORECASE)


def as_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in content)
    return str(content)


def normalize(messages) -> list[dict]:
    """Accept a str or a list of dicts/objects; return [{'role', 'content'}] with text content."""
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    out = []
    for m in messages or []:
        if isinstance(m, dict):
            out.append({"role": m.get("role", "user"), "content": as_text(m.get("content"))})
        else:
            out.append({"role": getattr(m, "role", "user"), "content": as_text(getattr(m, "content", m))})
    return out


def extract_marked(msgs: list[dict], tag: str):
    """Find [[TAG]]...[[/TAG]] in system/user messages (never in the model's own turns)."""
    pat = re.compile(rf"\[\[{tag}\]\](.*?)\[\[/{tag}\]\]", re.DOTALL)
    for m in msgs:
        if m["role"] == "assistant":
            continue
        hit = pat.search(m["content"])
        if hit:
            return hit.group(1).strip()
    return None


def tool_already_ran(msgs: list[dict]) -> bool:
    """True if the MODEL ITSELF (assistant turns only) already emitted an Action Input."""
    return any(m["role"] == "assistant" and "Action Input:" in m["content"] for m in msgs)


def latest_observation(msgs: list[dict]):
    """
    Tool result = text after the last 'Observation:' found from the model's OWN most recent tool-call
    turn onward. CrewAI may append the result to that assistant message, or send it as a later message
    (followed by an instruction like 'Analyze the tool result...'), so we scan that slice newest-first.
    The system prompt and task prompt come BEFORE this slice, so the template's example
    'Observation: the result of the action' can never be picked up.
    """
    start = max((i for i, m in enumerate(msgs)
                 if m["role"] == "assistant" and "Action Input:" in m["content"]), default=None)
    if start is None:
        return None
    for m in reversed(msgs[start:]):
        pos = m["content"].rfind("Observation:")
        if pos != -1:
            return m["content"][pos + len("Observation:"):].strip()
    return None


def schema_fields(tool) -> list[str]:
    schema = getattr(tool, "args_schema", None)
    fields = getattr(schema, "model_fields", None)
    return list(fields.keys()) if fields else []


def tool_kind(tool) -> str:
    """Classify by declared argument schema, never by tool name."""
    return "lookup" if "record_id" in schema_fields(tool) else "retrieval"


def build_tool_args(tool, question: str):
    fields = schema_fields(tool)
    if "record_id" in fields:
        hit = ORDER_ID_RE.search(question)
        return {"record_id": hit.group(0).upper()} if hit else None
    if fields:
        return {fields[0]: question}
    return None


def wrap_final(kind: str, payload: str) -> str:
    tag = "LOOKUP" if kind == "lookup" else "RETRIEVAL"
    return f"[{tag}]{payload}[/{tag}]"


def format_tool_call(tool_name: str, args: dict) -> str:
    return (f"Thought: I should use the {tool_name} tool to get what I need.\n"
            f"Action: {tool_name}\n"
            f"Action Input: {json.dumps(args)}")


def format_final(text: str) -> str:
    return f"Thought: I now know the final answer\nFinal Answer: {text}"


def decide(msgs: list[dict], tools: list) -> dict:
    """Return {'text': <what the mock model says>, 'decision': <label for the trace>, 'question': ...}."""
    question = extract_marked(msgs, "QUESTION")
    if question is None:
        first = next((m["content"] for m in msgs if m["role"] != "system"), "")
        question = first[:500]

    if not tools:
        return {"text": None, "decision": "no_tools", "question": question}

    if tool_already_ran(msgs):
        obs = latest_observation(msgs)
        if obs is None:                                  # never guess: make the problem visible
            obs = json.dumps({"error": "tool result not found in the conversation"})
        kind = tool_kind(tools[0])
        return {"text": format_final(wrap_final(kind, obs)), "decision": f"final_from_{kind}_observation",
                "question": question}

    for tool in tools:                                    # first tool whose arguments we can build
        args = build_tool_args(tool, question)
        if args is not None:
            name = getattr(tool, "name", "tool")
            return {"text": format_tool_call(name, args), "decision": f"tool_call:{name}:{tool_kind(tool)}",
                    "question": question}

    kind = tool_kind(tools[0])                            # e.g. lookup agent but the question has no order id
    payload = json.dumps({"found": False, "error": "No order id was found in the question."})
    return {"text": format_final(wrap_final(kind, payload)), "decision": "final_no_args", "question": question}