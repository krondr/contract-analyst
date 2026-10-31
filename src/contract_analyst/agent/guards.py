"""Step/time budget and repeated-call protection for the agent loop."""

from __future__ import annotations

import time

from contract_analyst.models import ToolResult


class Guard:
    def __init__(self, max_steps: int, max_seconds: int):
        self.max_steps, self.max_seconds = max_steps, max_seconds
        self.steps = 0
        self.started = time.monotonic()
        self._seen: set[tuple[str, str]] = set()

    def check_budget(self) -> ToolResult | None:
        if self.steps >= self.max_steps:
            return ToolResult(status="error", error_type="budget_exceeded",
                              hint=f"Лимит шагов ({self.max_steps}) исчерпан: заверши работу и переходи к отчёту")
        if time.monotonic() - self.started > self.max_seconds:
            return ToolResult(status="error", error_type="budget_exceeded",
                              hint=f"Лимит времени ({self.max_seconds} с) исчерпан: заверши работу и переходи к отчёту")
        return None

    def seen(self, key: tuple[str, str]) -> bool:
        if key in self._seen:
            return True
        self._seen.add(key)
        return False

    @property
    def exhausted(self) -> bool:
        return self.check_budget() is not None
