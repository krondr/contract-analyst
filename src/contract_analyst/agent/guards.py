"""Step/time budget, repeated-call protection and a cap on consecutive tool errors for the agent loop."""

from __future__ import annotations

import time

from contract_analyst.models import ToolResult


class Guard:
    def __init__(self, max_steps: int, max_seconds: int, max_errors_in_row: int = 5):
        self.max_steps, self.max_seconds, self.max_errors_in_row = max_steps, max_seconds, max_errors_in_row
        self.steps = 0
        self.errors_in_row = 0
        self.started = time.monotonic()
        self._seen: set[tuple] = set()

    def check_budget(self) -> ToolResult | None:
        if self.steps >= self.max_steps:
            return ToolResult(status="error", error_type="budget_exceeded",
                              hint=f"Лимит шагов ({self.max_steps}) исчерпан: заверши работу и переходи к отчёту")
        if time.monotonic() - self.started > self.max_seconds:
            return ToolResult(status="error", error_type="budget_exceeded",
                              hint=f"Лимит времени ({self.max_seconds} с) исчерпан: заверши работу и переходи к отчёту")
        return None

    def seen(self, key: tuple) -> bool:
        """Whether the key was seen before; it is remembered either way."""
        if key in self._seen:
            return True
        self._seen.add(key)
        return False

    def record(self, result: ToolResult) -> None:
        self.errors_in_row = self.errors_in_row + 1 if result.status == "error" else 0

    @property
    def stuck(self) -> bool:
        """The model keeps failing its calls: stop acting and let the reflector decide."""
        return self.errors_in_row >= self.max_errors_in_row

    @property
    def exhausted(self) -> bool:
        return self.check_budget() is not None
