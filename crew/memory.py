"""
Task 8 - session memory with LangChain's InMemoryChatMessageHistory + RunnableWithMessageHistory.

(RunnableWithMessageHistory raises a LangChainDeprecationWarning pointing to LangGraph persistence. That is
expected for this brief; the class still works, so the warning is not silenced.)

Memory is in-process only. Only MASKED text is ever stored, because masking happens before this layer.
"""
import re

from crew.mock_logic import ORDER_ID_RE

_STORE: dict = {}
_REFERS_TO_ORDER = re.compile(r"\b(it|its|this order|that order|my order|the order)\b", re.IGNORECASE)


def get_history(session_id: str):
    from langchain_core.chat_history import InMemoryChatMessageHistory     # lazy: keeps helpers testable
    if session_id not in _STORE:
        _STORE[session_id] = InMemoryChatMessageHistory()
    return _STORE[session_id]


def reset_all_sessions() -> None:
    _STORE.clear()


def build_chain(inner):
    """Wrap `inner(inputs: dict) -> dict` so history is loaded before and saved after each turn."""
    from langchain_core.runnables import RunnableLambda
    from langchain_core.runnables.history import RunnableWithMessageHistory
    return RunnableWithMessageHistory(
        RunnableLambda(inner), get_history,
        input_messages_key="input", history_messages_key="history", output_messages_key="output")


def resolve_references(question: str, history: list):
    """
    If the question has no order id but refers to 'it / my order', recover the most recent order id
    from the conversation history. Returns (effective_question, carried_order_id_or_None).
    """
    if ORDER_ID_RE.search(question):
        return question, None
    if _REFERS_TO_ORDER.search(question):
        for message in reversed(history):
            hit = ORDER_ID_RE.search(str(getattr(message, "content", "")))
            if hit:
                oid = hit.group(0).upper()
                return f"{question} (regarding order {oid})", oid
    return question, None