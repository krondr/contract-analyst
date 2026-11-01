"""JSONL trace of every agent step; thoughts are also echoed live to the console."""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path

from rich.console import Console

_LIVE = {"plan": "План", "thought": "Мысль", "reflection": "Рефлексия", "error": "Ошибка", "final": "Готово"}


class TraceLogger:
    def __init__(self, run_dir: Path, run_id: str, console: bool = True):
        run_dir.mkdir(parents=True, exist_ok=True)
        self.path = run_dir / "trace.jsonl"
        self.run_id = run_id
        self._console = Console(stderr=True) if console else None
        self._lock = threading.Lock()
        self._last_echo = None

    def event(self, kind: str, node: str, **payload) -> None:
        record = {"ts": datetime.now().isoformat(timespec="seconds"), "run_id": self.run_id, "node": node,
                  "event": kind, **payload}
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        if self._console and kind in _LIVE:
            text = payload.get("thought") or payload.get("message") or payload.get("items") or payload.get("summary") or ""
            tool = f" → {payload['tool']}" if "tool" in payload else ""
            # One thinking covers a whole batch of calls: print it once, then only the calls.
            shown = "(та же)" if text and (kind, text) == self._last_echo else text
            self._last_echo = (kind, text)
            self._console.print(f"[bold]{_LIVE[kind]}[/bold]{tool}: {shown}", highlight=False)


def read_trace(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def thought_counts(events: list[dict]) -> tuple[int, int]:
    """(tool calls with a thought from the argument, the message text or the thinking; all tool calls)."""
    sources = [e.get("thought_source") for e in events if e.get("event") == "thought"]
    return sum(s != "missing" for s in sources), len(sources)


def thought_coverage(events: list[dict]) -> float | None:
    found, total = thought_counts(events)
    return round(found / total, 3) if total else None
