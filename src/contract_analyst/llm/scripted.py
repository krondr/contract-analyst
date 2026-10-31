"""Deterministic scripted client for tests and offline demos."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from contract_analyst.llm.client import AssistantTurn, Message, new_usage


class ScriptExhausted(RuntimeError):
    pass


class ScriptedClient:
    model_name = "scripted"

    def __init__(self, script: list[AssistantTurn | str]):
        self._script = list(script)
        self.usage = new_usage()
        self.calls: list[dict] = []

    def _next(self) -> AssistantTurn | str:
        if not self._script:
            raise ScriptExhausted("scripted client has no more responses")
        self.usage["calls"] += 1
        return self._script.pop(0)

    def chat(self, messages: list[Message], tools: list[dict] | None = None, think: bool = False) -> AssistantTurn:
        self.calls.append({"kind": "chat", "messages": messages, "tools": tools, "think": think})
        item = self._next()
        return AssistantTurn(content=item) if isinstance(item, str) else item

    def structured(self, messages: list[Message], schema: type[BaseModel]) -> str:
        self.calls.append({"kind": "structured", "messages": messages, "schema": schema.__name__})
        item = self._next()
        return item.content if isinstance(item, AssistantTurn) else item


def load_script(path: Path) -> list[AssistantTurn | str]:
    """JSON list: {"structured": value} -> JSON text, {"chat": turn} -> AssistantTurn, "text" -> raw string."""
    items: list[AssistantTurn | str] = []
    for entry in json.loads(Path(path).read_text(encoding="utf-8")):
        if isinstance(entry, str):
            items.append(entry)
        elif "structured" in entry:
            items.append(json.dumps(entry["structured"], ensure_ascii=False))
        else:
            items.append(AssistantTurn.model_validate(entry["chat"]))
    return items
