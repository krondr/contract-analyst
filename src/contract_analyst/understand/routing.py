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
# A supplement states its operations in sub-clauses too («1. Внести в Договор следующие изменения: 1.1. ...»).
# Main contracts keep the narrower KEYWORDS["ops"] routing, so their recorded LLM answers stay valid.
_SUPPLEMENT_OPS = re.compile(
    r"изложить|редакци|вн(?:ести|осит\w*|осят\w*)\b.{0,40}изменени|изменить|заменить|дополнить|исключить"
    r"|утратившим\s+силу|недействительн|отменить|прекращ|прекрат|приостанов|расторг|вступает в силу|распространяет",
    re.IGNORECASE | re.DOTALL)
_FEW_CLAUSES = 2
# Where the parties sign in: «г. Москва «25» января 2021 г.» or «… в лице …». A contract that opens with «1. Определения»
# has it after the definitions, inside a clause the header batch would not read.
_SIGNING_IN = re.compile(r"\bг\.\s*[А-ЯЁ][а-яё-]+[\s,]*(?:[«\"]?\d{1,2}[»\"]?\s+[а-яё]+\s+\d{4}|\d{1,2}\.\d{2}\.\d{4})"
                        r"|\bв\s+лице\b")
_SIGNING_PAGES = 2
# Where a document states its own start or end, usually near its end: «Настоящее Соглашение вступает в силу с …»,
# «Договор действует до …», «Срок действия Договора …».
_OWN_TERM = re.compile(
    r"вступа\w*\s+в\s+(?:законную\s+)?силу"
    r"|(?:договор|соглашени|контракт)\w*[^.;]{0,80}?\b(?:действу\w*|заключ\w*)\s+(?:с|до|по|на\s+срок|сроком)\b"
    r"|срок\w*\s+действия\s+(?:настоящ\w+\s+)?(?:договор|соглашени|контракт)"
    # «Соглашение применяется к отношениям с …», «распространяет свое действие на отношения, возникшие с …»,
    # «Условия … соглашения вводятся с …»: the document says from when it applies.
    r"|(?:договор|соглашени|контракт)\w*[^.;]{0,80}?\b(?:примен\w*|распространя\w*|ввод\w*)\b",
    re.IGNORECASE)


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


def _quoted(text: str) -> list[tuple[int, int]]:
    """«…» pairs, where a supplement quotes new wording; an unmatched « opens nothing."""
    pairs, opened = [], []
    for i, ch in enumerate(text):
        if ch == "«":
            opened.append(i)
        elif ch == "»" and opened:
            pairs.append((opened.pop(), i))
    return pairs


def _states_own_term(text: str) -> bool:
    """The clause says when the document itself starts or ends, not only inside a quoted new wording of a clause."""
    quoted = _quoted(text)
    return any(not any(a < m.start() < b for a, b in quoted) for m in _OWN_TERM.finditer(text))


def _supplement_ops(doc: DocumentText) -> list[Clause]:
    if sum(c.clause_id != "preamble" for c in doc.clauses) <= _FEW_CLAUSES:
        return list(doc.clauses)  # (almost) unsegmented, e.g. Word auto-numbering: the whole short document
    quoted = _quoted("\n".join(c.text for c in doc.clauses))
    picked, offset = [], 0
    for clause in doc.clauses:
        hits = [offset + m.start() for m in _SUPPLEMENT_OPS.finditer(clause.text)]
        if any(not any(a < hit < b for a, b in quoted) for hit in hits):
            picked.append(clause)
        offset += len(clause.text) + 1
    return picked


def _chosen(doc: DocumentText, group: str, supplement: bool) -> list[Clause]:
    if group == "ops" and supplement:
        return _supplement_ops(doc)
    if group == "header":
        numbered = [c for c in doc.clauses if c.clause_id != "preamble"]
        first_page = doc.clauses[0].page if doc.clauses else 1
        signing = next((c for c in numbered[2:] if c.page < first_page + _SIGNING_PAGES and _SIGNING_IN.search(c.text)),
                       None)
        return [c for c in doc.clauses if c.clause_id == "preamble"] + numbered[:2] + ([signing] if signing else [])
    pattern = re.compile(KEYWORDS[group], re.IGNORECASE)
    picked = [c for c in doc.clauses if pattern.search(c.text)]
    if group == "ops":
        picked = [c for c in picked if _TOP_LEVEL.fullmatch(c.clause_id)]
    return picked


def select_clauses(doc: DocumentText, group: str, settings: Settings, supplement: bool = False) -> list[list[Clause]]:
    budget = settings.llm_money_batch_tokens if group == "money" else settings.llm_batch_tokens
    return _pack(_chosen(doc, group, supplement), budget)


def select_term_clauses(doc: DocumentText, settings: Settings) -> list[list[Clause]]:
    """Where the document states its own start or end, past the clauses the header batch already reads: a call of
    its own, so the batch that classifies the document stays the same."""
    numbered = [c for c in doc.clauses if c.clause_id != "preamble"]
    return _pack([c for c in numbered[2:] if _states_own_term(c.text)], settings.llm_batch_tokens)


def _pack(clauses: list[Clause], budget: int) -> list[list[Clause]]:
    batches: list[list[Clause]] = []
    current: list[Clause] = []
    for clause in clauses:
        for piece in _split(clause, budget * 3):
            if current and estimate_tokens([*current, piece]) > budget:
                batches.append(current)
                current = []
            current.append(piece)
    if current:
        batches.append(current)
    return batches
