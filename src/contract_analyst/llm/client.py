"""LLM client protocol shared by the Ollama, scripted and caching clients."""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel

Message = dict[str, Any]  # {"role": system|user|assistant|tool, "content": str, "tool_calls"?, "tool_call_id"?, "name"?}


class ToolCall(BaseModel):
    id: str
    name: str
    args: dict


class AssistantTurn(BaseModel):
    content: str = ""
    tool_calls: list[ToolCall] = []


class LLMUnavailable(RuntimeError):
    def __init__(self, hint: str):
        super().__init__(hint)
        self.hint = hint


class ReplayMiss(RuntimeError):
    pass


def new_usage() -> dict[str, int]:
    return {"calls": 0, "input_tokens": 0, "output_tokens": 0}


class LLMClient(Protocol):
    model_name: str
    usage: dict[str, int]

    def chat(self, messages: list[Message], tools: list[dict] | None = None) -> AssistantTurn: ...

    def structured(self, messages: list[Message], schema: type[BaseModel]) -> str: ...
