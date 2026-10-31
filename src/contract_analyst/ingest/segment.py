"""Split document text into numbered clauses and appendices."""

from __future__ import annotations

import re

from contract_analyst.models import Clause, Page

_CLAUSE_RE = re.compile(r"^\s*(\d{1,3}(?:\.\d{1,3}){1,3}|\d{1,3}(?=\.))\.?\s+\S")
_APPENDIX_RE = re.compile(r"^\s*Приложение\s*№\s*(\d+)", re.IGNORECASE)


def _clause_id(line: str) -> str | None:
    if m := _APPENDIX_RE.match(line):
        return f"Приложение {m.group(1)}"
    if m := _CLAUSE_RE.match(line):
        return m.group(1)
    return None


def segment_clauses(pages: list[Page]) -> list[Clause]:
    clauses: list[Clause] = []
    current_id, current_page, buffer = "preamble", pages[0].number if pages else 1, []

    def flush() -> None:
        text = "\n".join(buffer).strip()
        if text:
            clauses.append(Clause(clause_id=current_id, page=current_page, text=text))

    for page in pages:
        for line in page.text.split("\n"):
            new_id = _clause_id(line)
            if new_id:
                flush()
                current_id, current_page, buffer = new_id, page.number, []
            buffer.append(line)
    flush()
    return clauses
