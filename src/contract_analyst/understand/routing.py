"""Pick the clauses each field group needs and pack them into token-bounded batches."""

from __future__ import annotations

import math
import re

from contract_analyst.config import Settings
from contract_analyst.models import Clause, DocumentText

KEYWORDS = {
    "core_terms": r"предмет|срок|расторж|уведомл|реквизит|штраф|неустойк",
    "money": r"вознагражд|ставк|руб|оплат|ндс|цена|стоимост",
    "dealer": r"планов|показател|тарифн|торгов",
    "ops": r"изложить|внести изменени|дополнить|исключить|недействительн|прекращ|расторг|вступает в силу|распространяет",
}
_TOP_LEVEL = re.compile(r"\d{1,3}")


def estimate_tokens(clauses: list[Clause]) -> int:
    return math.ceil(sum(len(c.text) for c in clauses) / 3)


def _split(clause: Clause, max_chars: int) -> list[Clause]:
    parts, text = [], clause.text
    while len(text) > max_chars:
        cut = text.rfind(" ", 0, max_chars)
        cut = cut if cut > 0 else max_chars
        parts.append(clause.model_copy(update={"text": text[:cut]}))
        text = text[cut:]
    parts.append(clause.model_copy(update={"text": text}))
    return parts


def _chosen(doc: DocumentText, group: str) -> list[Clause]:
    if group == "header":
        numbered = [c for c in doc.clauses if c.clause_id != "preamble"][:2]
        return [c for c in doc.clauses if c.clause_id == "preamble"] + numbered
    pattern = re.compile(KEYWORDS[group], re.IGNORECASE)
    picked = [c for c in doc.clauses if pattern.search(c.text)]
    if group == "ops":
        picked = [c for c in picked if _TOP_LEVEL.fullmatch(c.clause_id)]
    return picked


def select_clauses(doc: DocumentText, group: str, settings: Settings) -> list[list[Clause]]:
    budget = settings.llm_batch_tokens
    batches: list[list[Clause]] = []
    current: list[Clause] = []
    for clause in _chosen(doc, group):
        for piece in _split(clause, budget * 3):
            if current and estimate_tokens([*current, piece]) > budget:
                batches.append(current)
                current = []
            current.append(piece)
    if current:
        batches.append(current)
    return batches
