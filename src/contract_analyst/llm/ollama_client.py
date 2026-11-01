"""Local Ollama client (langchain-ollama). Every call is capped in output tokens and in wall-clock time. Calls
without thinking are greedy; Qwen3 thinking turns use the sampling Qwen3 is made for."""

from __future__ import annotations

import ipaddress
import os
import time
from urllib.parse import urlsplit

import httpx
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage, ToolMessage
from ollama import ResponseError
from pydantic import BaseModel

from contract_analyst.llm.client import AssistantTurn, LLMUnavailable, Message, ToolCall, new_usage

# Qwen3 model card: thinking needs temperature 0.6, top_p 0.95, top_k 20, min_p 0; greedy decoding degrades it and
# can repeat endlessly. Replays stay deterministic: the cache records every turn.
THINKING_SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0}
GREEDY = {"temperature": 0.0}
# The connection or the server failed (not our code): callers record it and the run goes on.
TRANSPORT_ERRORS = (TimeoutError, ConnectionError, httpx.TransportError, ResponseError)


def is_loopback(url: str) -> bool:
    """By the address as written: no DNS lookup, which could itself leave the machine or be spoofed."""
    host = urlsplit(url if "://" in url else f"http://{url}").hostname or ""
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


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
                 timeout: float = 600.0, max_tokens: int = 4096, think_max_tokens: int = 4096,
                 keep_alive: str = "30m", deadline: float = 900.0):
        if not is_loopback(base_url) and os.environ.get("CA_ALLOW_REMOTE_LLM") != "1":
            raise LLMUnavailable(f"Модель по адресу {base_url} — не на этой машине: тексты документов ушли бы на "
                                 "этот сервер. Если сервер ваш и это допустимо, задайте CA_ALLOW_REMOTE_LLM=1.")
        self.model_name, self.base_url, self.num_ctx, self.timeout = model, base_url, num_ctx, timeout
        self.max_tokens, self.think_max_tokens = max_tokens, think_max_tokens
        self.keep_alive, self.deadline = keep_alive, deadline
        self.usage = new_usage() | {"truncated": 0}  # truncated: answers cut at the token cap
        self._llms: dict[bool, object] = {}

    def _get(self, think: bool = False):
        if think not in self._llms:
            from langchain_ollama import ChatOllama

            # The read timeout ends a connection that goes silent; the deadline ends one that never stops streaming.
            self._llms[think] = ChatOllama(model=self.model_name, base_url=self.base_url, reasoning=think,
                                           keep_alive=self.keep_alive,
                                           client_kwargs={"timeout": min(self.timeout, self.deadline)})
        return self._llms[think]

    def _options(self, think: bool) -> dict:
        cap = self.think_max_tokens if think else self.max_tokens
        return {"num_ctx": self.num_ctx, "num_predict": cap, **(THINKING_SAMPLING if think else GREEDY)}

    def check_available(self) -> None:
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=2.0)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMUnavailable(f"Ollama не отвечает по адресу {self.base_url}: запустите `ollama serve`") from exc
        names = {m.get("name") for m in response.json().get("models", [])}
        if self.model_name not in names and f"{self.model_name}:latest" not in names:
            raise LLMUnavailable(f"Модель {self.model_name} не скачана: выполните `ollama pull {self.model_name}`")

    def _complete(self, runnable, messages: list[Message], think: bool) -> AIMessageChunk:
        """Stream the answer and stop at the deadline. Closing the stream closes the connection, and that is what
        makes Ollama abort the generation."""
        deadline = time.monotonic() + self.deadline
        stream = runnable.stream(_to_langchain(messages), options=self._options(think))
        ai = AIMessageChunk(content="")
        try:
            for chunk in stream:
                ai = ai + chunk
                if time.monotonic() > deadline:
                    raise TimeoutError(f"ответ модели не уложился в {self.deadline:.0f} с: запрос прерван")
        finally:
            stream.close()
        self.usage["calls"] += 1
        meta = ai.usage_metadata or {}
        self.usage["input_tokens"] += meta.get("input_tokens", 0)
        self.usage["output_tokens"] += meta.get("output_tokens", 0)
        self.usage["truncated"] += ai.response_metadata.get("done_reason") == "length"
        return ai

    def chat(self, messages: list[Message], tools: list[dict] | None = None, think: bool = False) -> AssistantTurn:
        llm = self._get(think)
        ai = self._complete(llm.bind_tools(tools) if tools else llm, messages, think)
        calls = [ToolCall(id=c.get("id") or f"call_{i}", name=c["name"], args=c.get("args") or {})
                 for i, c in enumerate(ai.tool_calls)]
        return AssistantTurn(content=_text(ai.content), tool_calls=calls,
                             reasoning=ai.additional_kwargs.get("reasoning_content") or "")

    def structured(self, messages: list[Message], schema: type[BaseModel]) -> str:
        return _text(self._complete(self._get().bind(format=schema.model_json_schema()), messages, False).content)
