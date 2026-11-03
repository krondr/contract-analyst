"""Response cache around any client: record, replay (fail on miss) or off."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from contract_analyst.llm.client import AssistantTurn, LLMClient, Message, ReplayMiss


class CachingClient:
    def __init__(self, inner: LLMClient, db_path: Path, mode: Literal["record", "replay", "off"]):
        self.inner, self.mode = inner, mode
        self.model_name = inner.model_name
        self.usage = inner.usage
        self.hits = 0
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, kind TEXT, payload TEXT)")
        self._lock = threading.Lock()

    def check_available(self) -> None:
        if self.mode != "replay" and hasattr(self.inner, "check_available"):
            self.inner.check_available()

    def cache_params(self) -> dict:
        """The inner client's generation parameters (caps, context, sampling); a client without them has none."""
        return self.inner.cache_params() if hasattr(self.inner, "cache_params") else {}

    def _key(self, kind: str, messages: list[Message], tools: list[dict] | None, schema: type[BaseModel] | None) -> str:
        payload = {"model": self.model_name, "kind": kind, "messages": messages, "tools": tools,
                   "schema": schema.model_json_schema() if schema else None, "params": self.cache_params()}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()

    def _get(self, key: str) -> str | None:
        with self._lock:
            row = self._db.execute("SELECT payload FROM responses WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def _put(self, key: str, kind: str, payload: str) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO responses VALUES (?, ?, ?)", (key, kind, payload))
            self._db.commit()

    def _lookup(self, key: str) -> str | None:
        cached = self._get(key)
        if cached is not None:
            self.hits += 1
            return cached
        if self.mode == "replay":
            raise ReplayMiss(f"no cached LLM response for key {key[:12]} (replay mode)")
        return None

    def _truncated(self) -> int:
        """Answers the inner client has cut at the token cap so far."""
        return self.inner.usage.get("truncated", 0)

    def chat(self, messages: list[Message], tools: list[dict] | None = None, think: bool = False) -> AssistantTurn:
        if self.mode == "off":
            return self.inner.chat(messages, tools, think=think)
        key = self._key("chat+think" if think else "chat", messages, tools, None)
        if (cached := self._lookup(key)) is not None:
            return AssistantTurn.model_validate_json(cached)
        cut = self._truncated()
        turn = self.inner.chat(messages, tools, think=think)
        if self._truncated() == cut:  # an answer cut at the token cap is not kept: the next run asks again
            self._put(key, "chat", turn.model_dump_json())
        return turn

    def structured(self, messages: list[Message], schema: type[BaseModel]) -> str:
        if self.mode == "off":
            return self.inner.structured(messages, schema)
        key = self._key("structured", messages, None, schema)
        if (cached := self._lookup(key)) is not None:
            return cached
        cut = self._truncated()
        raw = self.inner.structured(messages, schema)
        if self._truncated() == cut:
            self._put(key, "structured", raw)
        return raw
