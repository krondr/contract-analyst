"""Local Ollama client (langchain-ollama), temperature 0; Qwen3 thinking only when a caller asks for it."""

from __future__ import annotations

import httpx
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel

from contract_analyst.llm.client import AssistantTurn, LLMUnavailable, Message, ToolCall, new_usage


def _to_langchain(messages: list[Message]) -> list:
    converted = []
    for m in messages:
        role, content = m["role"], m.get("content") or ""
        if role == "system":
            converted.append(SystemMessage(content))
        elif role == "user":
            converted.append(HumanMessage(content))
        elif role == "assistant":
            calls = [{"name": c["name"], "args": c["args"], "id": c["id"]} for c in m.get("tool_calls") or []]
            converted.append(AIMessage(content, tool_calls=calls))
        elif role == "tool":
            converted.append(ToolMessage(content, tool_call_id=m["tool_call_id"]))
    return converted


def _text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)


class OllamaClient:
    def __init__(self, model: str, base_url: str = "http://localhost:11434", num_ctx: int = 16384,
                 timeout: float = 600.0):
        self.model_name, self.base_url, self.num_ctx, self.timeout = model, base_url, num_ctx, timeout
        self.usage = new_usage()
        self._llms: dict[bool, object] = {}

    def _get(self, think: bool = False):
        if think not in self._llms:
            from langchain_ollama import ChatOllama

            self._llms[think] = ChatOllama(model=self.model_name, base_url=self.base_url, temperature=0,
                                           num_ctx=self.num_ctx, reasoning=think,
                                           client_kwargs={"timeout": self.timeout})
        return self._llms[think]

    def check_available(self) -> None:
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=2.0)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMUnavailable(f"Ollama не отвечает по адресу {self.base_url}: запустите `ollama serve`") from exc
        names = {m.get("name") for m in response.json().get("models", [])}
        if self.model_name not in names and f"{self.model_name}:latest" not in names:
            raise LLMUnavailable(f"Модель {self.model_name} не скачана: выполните `ollama pull {self.model_name}`")

    def _account(self, ai: AIMessage) -> None:
        self.usage["calls"] += 1
        meta = ai.usage_metadata or {}
        self.usage["input_tokens"] += meta.get("input_tokens", 0)
        self.usage["output_tokens"] += meta.get("output_tokens", 0)

    def chat(self, messages: list[Message], tools: list[dict] | None = None, think: bool = False) -> AssistantTurn:
        llm = self._get(think)
        ai = (llm.bind_tools(tools) if tools else llm).invoke(_to_langchain(messages))
        self._account(ai)
        calls = [ToolCall(id=c.get("id") or f"call_{i}", name=c["name"], args=c.get("args") or {})
                 for i, c in enumerate(ai.tool_calls)]
        return AssistantTurn(content=_text(ai.content), tool_calls=calls,
                             reasoning=ai.additional_kwargs.get("reasoning_content") or "")

    def structured(self, messages: list[Message], schema: type[BaseModel]) -> str:
        ai = self._get().bind(format=schema.model_json_schema()).invoke(_to_langchain(messages))
        self._account(ai)
        return _text(ai.content)
