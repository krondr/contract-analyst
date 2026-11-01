"""Code-level check of LLM-written text against the sources it cites.

A sentence survives only if each of its references exists and each number, date and document number in it is found
on a cited page or in a knowledge-base value taken from that page.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from datetime import date
from decimal import Decimal
from functools import lru_cache

from pydantic import BaseModel

from contract_analyst.models import (
    AmendmentOp,
    ConsolidatedContract,
    ContractChain,
    DocumentRecord,
    Extracted,
    RateRow,
    Source,
    TermChange,
    ValueStatus,
)
from contract_analyst.parsing.patterns import DOC_NUMBER_RE, find_doc_numbers

_BLOCK = re.compile(r"<!-- llm( refs)? -->(.*?)<!-- /llm -->", re.DOTALL)
_REF = re.compile(r"\[doc:[^\]\s]+\]")
# A possible sentence end: terminal punctuation with the closing quotes and references that belong to the
# sentence, then whitespace; or a line break. A period inside "3.1" or "01.04.2025" is not followed by a space.
_END = re.compile(r"(?P<mark>[.!?…]+)[»\"”)]*(?:\s*\[doc:[^\]\s]+\])*(?P<close>[.!?…]?)(?=\s|$)|\n")
# Abbreviations after which a sentence goes on even before a capital: "ст. 450 ГК", "ул. Ленина", "т.е. Договор".
# Single letters are handled apart: initials, "п.", "ч.", "г. Тверь".
_INNER = {"пп", "ст", "гл", "разд", "подп", "абз", "прил", "см", "им", "ул", "обл", "пр", "пер", "кв", "стр", "т.е",
          "т.ч", "т.к", "напр", "тел", "рис", "табл", "ред", "№"}

_MONTHS = ("января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября",
           "ноября", "декабря")
_DATE = re.compile(r"(?<!\d)(?:(\d{1,2})\.(\d{1,2})\.(\d{4})|(\d{4})-(\d{2})-(\d{2})"
                   rf"|«?(\d{{1,2}})»?\s+({'|'.join(_MONTHS)})\b(?:\s+(\d{{4}}))?)(?!\d)", re.IGNORECASE)
_NUMBER = re.compile(r"(?<![\w.,])(\d{1,3}(?:[   ]\d{3})+|\d+)((?:[.,]\d+)*)(?:\s*(тыс|млн|млрд)\b)?",
                     re.IGNORECASE)
_SCALE = {"тыс": 1000, "млн": 10**6, "млрд": 10**9}
_FRAME = re.compile(r"\bна\s+(?:дату\s+)?$", re.IGNORECASE)
_KEEP = (ValueStatus.ACCEPTED, ValueStatus.CONFLICT)


# --- sentences ---------------------------------------------------------------------------------------------
def _ends(text: str, start: int, m: re.Match[str]) -> bool:
    if m.group(0) == "\n":
        return True
    following = text[m.end():].lstrip().lstrip("«\"„“*")
    if not following:
        return True
    if not following[0].isupper():  # "руб. в месяц", "п. 3.1", "т.е. 1 500"
        return False
    if m.group("mark") != "." or m.group("close"):
        return True
    words = text[start:m.start()].split()
    word = words[-1].lstrip("«\"(").lower() if words else ""
    if word in _INNER or (word.isdigit() and len(words) == 1):  # list numbering: "1. Ставка ..."
        return False
    if len(word) == 1 and word.isalpha():  # "г." ends a sentence only as "год": "в 2025 г. Срок ..."
        return word == "г" and len(words) > 1 and words[-2][-1].isdigit()
    return True


def split_sentences(text: str) -> list[str]:
    """Split into sentences without losing characters: "".join(parts) == text."""
    parts, start = [], 0
    for m in _END.finditer(text):
        if m.start() < start or not _ends(text, start, m):
            continue
        end = len(text) - len(text[m.end():].lstrip())
        parts.append(text[start:end])
        start = end
    return parts + [text[start:]] if start < len(text) else parts


# --- facts a sentence states, facts a source backs --------------------------------------------------------
def _date_facts(m: re.Match[str], backing: bool) -> set[str]:
    d, mo, y, iy, im, iday, td, tm, ty = m.groups()
    if iy:
        y, mo, d = iy, im, iday
    elif td:
        d, mo, y = td, _MONTHS.index(tm.lower()) + 1, ty
    try:
        day = date(int(y or 2000), int(mo), int(d))  # a leap year stands in when only day and month are given
    except ValueError:
        return set()
    if not y:
        return {f"dm:{day:%m-%d}"}
    return {f"date:{day}", f"dm:{day:%m-%d}", f"num:{day.year}"} if backing else {f"date:{day}"}


def _number_facts(m: re.Match[str], backing: bool) -> set[str]:
    whole, frac, scale = m.groups()
    digits = re.sub(r"\D", "", whole)
    if frac.count(".") + frac.count(",") > 1:  # clause numbers like 2.1.1
        return {f"num:{digits}{frac}"}
    value = Decimal(digits + frac.replace(",", "."))
    found = {value * _SCALE[scale.lower()] if scale else value}
    if backing:  # "120 тыс." also backs 120; "270 500" may be two table cells
        found |= {value, *(Decimal(group) for group in whole.split())}
    return {"num:" + format(v.normalize(), "f") for v in found}


@lru_cache(maxsize=4096)
def _facts(text: str, backing: bool = False) -> frozenset[str]:
    """Document numbers, dates and numbers, normalized so that 01.04.2025, «1» апреля 2025 г. and 2025-04-01 match,
    and so do 120 000, 120000,00 and 120 тыс. Backing text also yields years of its dates and OCR-misread numbers."""
    numbers = DOC_NUMBER_RE.findall(text) + (find_doc_numbers(text) if backing else [])
    found = {f"no:{n}" for n in numbers} | ({f"no:{n.split('-')[0]}" for n in numbers} if backing else set())
    text = DOC_NUMBER_RE.sub(" ; ", text)
    for m in _DATE.finditer(text):
        found |= _date_facts(m, backing)
    for m in _NUMBER.finditer(_DATE.sub(" ; ", text)):
        found |= _number_facts(m, backing)
    return frozenset(found)


# --- what each reference can back ---------------------------------------------------------------------------
def _plain(value: object) -> str:
    if isinstance(value, BaseModel):
        return " ".join(_plain(v) for v in value.__dict__.values())
    if isinstance(value, Decimal):
        return format(value.normalize(), "f")
    return "" if value is None else str(value)


def _cited_value(obj: BaseModel) -> tuple[Source, list] | None:
    if isinstance(obj, Extracted) and obj.status in _KEEP:
        return obj.source, [obj.value]
    if isinstance(obj, RateRow) and obj.status == ValueStatus.ACCEPTED:
        return obj.source, [obj.label, obj.amount, obj.valid_from, obj.valid_to]
    if isinstance(obj, AmendmentOp) and obj.status == ValueStatus.ACCEPTED:
        return obj.source, [obj.target, obj.new_text, obj.effective_from, obj.effective_to]
    if isinstance(obj, TermChange):
        return obj.source, [obj.value, obj.effective_from]
    return None


def _collect(obj: object, out: dict[tuple[str, int], list[str]]) -> None:
    if isinstance(obj, dict):
        obj = list(obj.values())
    if isinstance(obj, list):
        for item in obj:
            _collect(item, out)
    elif isinstance(obj, BaseModel):
        cited = _cited_value(obj)
        if cited and cited[0] is not None:
            out.setdefault((cited[0].doc_id, cited[0].page), []).append(" ".join(_plain(v) for v in cited[1]))
        for name in type(obj).model_fields:
            _collect(getattr(obj, name), out)


def grounding(pages: Iterable[tuple[str, int, str]], records: dict[str, DocumentRecord],
              chains: list[ContractChain], consolidated: dict[str, ConsolidatedContract]) -> dict[str, str]:
    """Every page reference with the text that can back a claim citing it: the page text, the accepted and
    conflicting values taken from that page, and the document numbers of the chain the page belongs to."""
    values: dict[tuple[str, int], list[str]] = {}
    _collect([records, chains, consolidated], values)
    chain_numbers = {d.doc_id: " ".join(n for n in [c.contract_number, *(x.number for x in c.docs)] if n)
                     for c in chains for d in c.docs}
    return {f"doc:{doc_id}:p{page}": "\n".join([text, chain_numbers.get(doc_id, ""), *values.get((doc_id, page), [])])
            for doc_id, page, text in pages}


# --- the check ------------------------------------------------------------------------------------------------
def _unframed(sentence: str, asked: frozenset[str]) -> str:
    """Drop "на 30.10.2026" when the user asked about that date: it frames the answer and is not a claim."""
    def drop(m: re.Match[str]) -> str:
        framing = _date_facts(m, backing=False) & asked and _FRAME.search(sentence[:m.start()])
        return " " if framing else m.group(0)

    return _DATE.sub(drop, sentence) if asked else sentence


def _supported(sentence: str, sources: Mapping[str, str], require_refs: bool, asked: frozenset[str]) -> bool:
    refs = [r[1:-1] for r in _REF.findall(sentence)]
    if any(r not in sources for r in refs):
        return False
    claims = _facts(_unframed(_REF.sub(" ", sentence), asked))
    if not refs:
        return not claims and not (require_refs and re.search(r"\w", sentence))
    return claims <= frozenset().union(*(_facts(sources[r], backing=True) for r in refs))


def verify_text(text: str, sources: Mapping[str, str], require_refs: bool = False,
                question: str = "") -> tuple[str, int]:
    """Drop the sentences the cited sources do not back; return the rest and how many were dropped.

    require_refs: every statement needs a reference, not only one with numbers (chat answers).
    question: a date the user asked about may frame the answer ("на 30.10.2026 ...") without a source.
    """
    asked = frozenset(f for f in _facts(question) if f.startswith("date:"))
    kept, removed = [], 0
    for part in split_sentences(text):
        if not part.strip() or _supported(part.strip(), sources, require_refs, asked):
            kept.append(part)
        else:
            removed += 1
    return "".join(kept).strip(), removed


def verify_claims(markdown: str, sources: Mapping[str, str]) -> tuple[str, int]:
    """Check every <!-- llm --> block of a rendered report; sources come from grounding(). A <!-- llm refs --> block
    states facts about the contracts, so a sentence without a reference goes too (spec §6.3)."""
    total = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal total
        text, removed = verify_text(match.group(2), sources, require_refs=bool(match.group(1)))
        total += removed
        return f"<!-- llm{match.group(1) or ''} -->{text}<!-- /llm -->"

    return _BLOCK.sub(replace, markdown), total
