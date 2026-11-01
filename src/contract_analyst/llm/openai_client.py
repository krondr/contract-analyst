"""OpenAI-compatible chat server inside the customer's perimeter (vLLM, LM Studio, llama.cpp server, TGI).

Plain HTTP on /chat/completions, so no vendor SDK: tools in the OpenAI format the agent already uses, structured
answers through response_format json_schema. Same guarantees as the Ollama client: a non-loopback server needs
CA_ALLOW_REMOTE_LLM=1, every call is capped in tokens and in time, cut answers are counted."""

from __future__ import annotations

import json
import os

import httpx
from pydantic import BaseModel

from contract_analyst.llm.client import AssistantTurn, LLMUnavailable, Message, ToolCall, new_usage
from contract_analyst.llm.ollama_client import is_loopback

GREEDY = {"temperature": 0}
THINKING = {"temperature": 0.6, "top_p": 0.95}  # whether the model thinks is the server's chat-template setting


def _to_openai(messages: list[Message]) -> list[dict]:
    converted = []
    for m in messages:
        if m["role"] == "assistant" and m.get("tool_calls"):
            converted.append({"role": "assistant", "content": m.get("content") or "", "tool_calls": [
                {"id": c["id"], "type": "function",
                 "function": {"name": c["name"], "arguments": json.dumps(c["args"], ensure_ascii=False)}}
                for c in m["tool_calls"]]})
        elif m["role"] == "tool":
            converted.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m.get("content") or ""})
        else:
            converted.append({"role": m["role"], "content": m.get("content") or ""})
    return converted


def _closed(schema: object) -> object:
    """The schema with additionalProperties: false on every object that does not set it: OpenAI rejects
    response_format schemas without it. A map (dict field) keeps the schema of its values."""
    if isinstance(schema, dict):
        node = {k: _closed(v) for k, v in schema.items()}
        if (node.get("type") == "object" or "properties" in node) and "additionalProperties" not in node:
            node["additionalProperties"] = False
        return node
    if isinstance(schema, list):
        return [_closed(v) for v in schema]
    return schema


def _args(raw: str | dict | None) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}  # the tool then reports bad arguments to the agent
    return parsed if isinstance(parsed, dict) else {}


class OpenAICompatClient:
    def __init__(self, model: str, base_url: str = "http://localhost:8000/v1", api_key: str | None = None,
                 max_tokens: int = 4096, think_max_tokens: int = 4096, deadline: float = 900.0,
                 extra_body: dict | None = None, think_extra_body: dict | None = None,
                 transport: httpx.BaseTransport | None = None):
        if not is_loopback(base_url) and os.environ.get("CA_ALLOW_REMOTE_LLM") != "1":
            raise LLMUnavailable(f"Модель по адресу {base_url} — не на этой машине: тексты документов ушли бы на "
                                 "этот сервер. Если сервер ваш и это допустимо, задайте CA_ALLOW_REMOTE_LLM=1.")
        self.model, self.base_url = model, base_url.rstrip("/")
        self.model_name = f"openai:{model}"  # also the cache key: never shared with an Ollama model of that name
        self.max_tokens, self.think_max_tokens = max_tokens, think_max_tokens
        # Server-specific fields, e.g. switching a reasoning model's thinking off for extraction and on for the agent.
        self.extra_body, self.think_extra_body = extra_body or {}, think_extra_body or {}
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        # A non-streamed answer arrives whole, so the read timeout is the per-call deadline.
        self._http = httpx.Client(base_url=self.base_url, headers=headers, timeout=deadline, transport=transport)
        self.usage = new_usage() | {"truncated": 0}

    def check_available(self) -> None:
        try:
            response = self._http.get("/models", timeout=5.0)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in (401, 403):
                raise LLMUnavailable(f"Сервер {self.base_url} требует ключ: задайте CA_OPENAI_API_KEY") from exc
            raise LLMUnavailable(f"Сервер модели {self.base_url} ответил {exc.response.status_code}") from exc
        except httpx.HTTPError as exc:
            raise LLMUnavailable(f"Сервер модели не отвечает по адресу {self.base_url}: запустите его "
                                 "или задайте CA_OPENAI_BASE_URL") from exc
        ids = sorted(m.get("id") for m in response.json().get("data", []))
        if self.model not in ids:
            raise LLMUnavailable(f"Модели {self.model} нет на сервере {self.base_url}; есть: {', '.join(ids) or '—'}")

    def _complete(self, body: dict) -> dict:
        response = self._http.post("/chat/completions", json={"model": self.model, **body})
        if response.is_error:  # the server failed, not our code: the run records it and goes on
            raise ConnectionError(f"сервер модели ответил {response.status_code}: {response.text[:200]}")
        data = response.json()
        usage = data.get("usage") or {}
        self.usage["calls"] += 1
        self.usage["input_tokens"] += usage.get("prompt_tokens", 0)
        self.usage["output_tokens"] += usage.get("completion_tokens", 0)
        choice = data["choices"][0]
        self.usage["truncated"] += choice.get("finish_reason") == "length"
        return choice["message"]

    def chat(self, messages: list[Message], tools: list[dict] | None = None, think: bool = False) -> AssistantTurn:
        body = {"messages": _to_openai(messages), "max_tokens": self.think_max_tokens if think else self.max_tokens,
                **(THINKING if think else GREEDY), **(self.think_extra_body if think else self.extra_body)}
        if tools:
            body["tools"] = tools
        message = self._complete(body)
        calls = [ToolCall(id=c.get("id") or f"call_{i}", name=c["function"]["name"], args=_args(c["function"].get("arguments")))
                 for i, c in enumerate(message.get("tool_calls") or [])]
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        return AssistantTurn(content=message.get("content") or "", tool_calls=calls, reasoning=reasoning)

    def structured(self, messages: list[Message], schema: type[BaseModel]) -> str:
        message = self._complete({"messages": _to_openai(messages), "max_tokens": self.max_tokens, **GREEDY,
                                  **self.extra_body,
                                  "response_format": {"type": "json_schema", "json_schema": {
                                      "name": schema.__name__, "schema": _closed(schema.model_json_schema()),
                                      "strict": False}}})
        return message.get("content") or ""
