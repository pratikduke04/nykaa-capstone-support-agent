"""
Task 14 - MockChatClient: a deterministic, keyless model for Autogen (MOCK_LLM mode).

Extends autogen_core's ChatCompletionClient (the 9 abstract members found by the API probe). Each agent
announces its role in its system message ([[ROLE:REVIEWER]] / [[ROLE:EDITOR]]); the client reads the role and the
conversation it is given and returns what a model would: a fact-check report, or a Verdict as JSON text.
Token counts are word counts, so the governance budget (Task 15) has real numbers to work with.
"""
import json
import re
from typing import Any, AsyncGenerator

from autogen_core.models import ChatCompletionClient, CreateResult, ModelInfo, RequestUsage

from review import review_logic as rl


def _text(message) -> str:
    content = getattr(message, "content", "")
    return content if isinstance(content, str) else json.dumps(content, default=str)


def _words(text: str) -> int:
    return len(text.split())


class MockChatClient(ChatCompletionClient):
    def __init__(self):
        self.calls: list[str] = []                       # one role name per model call
        self._total = RequestUsage(prompt_tokens=0, completion_tokens=0)
        self._last = RequestUsage(prompt_tokens=0, completion_tokens=0)

    # ---- the model itself ----
    def _respond(self, messages) -> str:
        everything = "\n".join(_text(m) for m in messages)
        role = re.search(r"\[\[ROLE:(\w+)\]\]", everything)
        role = role.group(1) if role else "UNKNOWN"
        self.calls.append(role)
        if role == "REVIEWER":
            return rl.reviewer_report(everything)
        if role == "EDITOR":
            return json.dumps(rl.editor_verdict(everything))
        return json.dumps({"error": "unknown role"})

    async def create(self, messages, **kwargs) -> CreateResult:
        out = self._respond(messages)
        prompt = sum(_words(_text(m)) for m in messages)
        self._last = RequestUsage(prompt_tokens=prompt, completion_tokens=_words(out))
        self._total = RequestUsage(prompt_tokens=self._total.prompt_tokens + prompt,
                                   completion_tokens=self._total.completion_tokens + self._last.completion_tokens)
        return CreateResult(finish_reason="stop", content=out, usage=self._last, cached=False)

    async def create_stream(self, messages, **kwargs) -> AsyncGenerator[Any, None]:
        yield await self.create(messages, **kwargs)

    # ---- bookkeeping the interface requires ----
    async def close(self) -> None:
        return None

    def actual_usage(self) -> RequestUsage:
        return self._last

    def total_usage(self) -> RequestUsage:
        return self._total

    def count_tokens(self, messages, **kwargs) -> int:
        return sum(_words(_text(m)) for m in messages)

    def remaining_tokens(self, messages, **kwargs) -> int:
        return 128000 - self.count_tokens(messages)

    @property
    def capabilities(self) -> dict:
        return {"vision": False, "function_calling": False, "json_output": True}

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(vision=False, function_calling=False, json_output=True,
                         family="unknown", structured_output=True)