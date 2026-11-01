"""Deterministic first-page header parsing: title kind, number, date, city, parent reference."""

from __future__ import annotations

import re
from datetime import date
from typing import Literal

from pydantic import BaseModel

from contract_analyst.ingest.normalize import normalize_text
from contract_analyst.parsing.patterns import _OCR_NUMBER_RE, find_dates, find_doc_numbers

_TITLE_RE = re.compile(r"^\s*(?:(?!дополнительн)[А-ЯЁа-яё-]+(?:ий|ый|ой|ая|ое)\s+)?(дополнительное\s+соглашение|соглашение|договор)\b",
                       re.IGNORECASE)
_TITLE_KIND = {"дополнительное": "supplement", "соглашение": "agreement", "договор": "contract"}
_CITY_RE = re.compile(r"^\s*г\.\s*([А-ЯЁ][а-яё-]+)")
# «к Агентскому договору»: up to two adjectives between the preposition and the noun.
_PARENT_ANCHOR_RE = re.compile(r"\b(?:к|в|о\s+расторжении)\s+(?:[а-яё-]+(?:ому|ему|ой|ым)\s+){0,2}"
                               r"(?:договор\w*|соглашени\w*)", re.IGNORECASE)
_DRAFT_LINE_RE = re.compile(r"^\s*проект(?:\s+договора)?\s*[.:]?\s*$", re.IGNORECASE)
_DRAFT_NAME_RE = re.compile(r"(?<![а-яё])проект(?![а-яё])")
_HEADER_LINES = 15
_PARENT_WINDOW = 200


class HeaderGuess(BaseModel):
    title_kind: Literal["contract", "supplement", "agreement", "unknown"] = "unknown"
    number: str | None = None
    doc_date: date | None = None
    city: str | None = None
    parent_number: str | None = None
    parent_date: date | None = None
    spans: dict[str, str] = {}


def _parent_reference(text: str) -> tuple[str, date, str] | None:
    for anchor in _PARENT_ANCHOR_RE.finditer(text):
        window = text[anchor.end(): anchor.end() + _PARENT_WINDOW]
        numbers = list(_OCR_NUMBER_RE.finditer(window))
        dates = [(window.find(span), d, span) for d, span in find_dates(window)]
        if not numbers or not dates:
            continue
        num = numbers[0]
        pos, d, span = min(dates, key=lambda item: abs(item[0] - num.start()))
        return f"D{num.group(1)}", d, text[anchor.start(): anchor.end() + max(num.end(), pos + len(span))]
    return None


def parse_header(first_page_text: str) -> HeaderGuess:
    lines = [ln for ln in normalize_text(first_page_text).split("\n")][:_HEADER_LINES]
    guess = HeaderGuess()
    title_idx = next((i for i, ln in enumerate(lines) if _TITLE_RE.match(ln)), None)
    if title_idx is None:
        return guess
    title = lines[title_idx]
    guess.title_kind = _TITLE_KIND[_TITLE_RE.match(title).group(1).split()[0].lower()]
    numbers = find_doc_numbers(title)
    if numbers:
        guess.number = numbers[0]
        guess.spans["number"] = title

    parent = _parent_reference(" ".join(lines[title_idx: title_idx + 4]))
    if parent:
        guess.parent_number, guess.parent_date, guess.spans["parent"] = parent

    city_lines = [ln for ln in lines[title_idx:] if _CITY_RE.match(ln)] or [ln for ln in lines if _CITY_RE.match(ln)]
    if city_lines:
        city_line = city_lines[0]
        guess.city = _CITY_RE.match(city_line).group(1)
        guess.spans["city"] = city_line
        dates = find_dates(city_line)
        if dates:
            guess.doc_date = dates[0][0]
            guess.spans["doc_date"] = city_line
    return guess


def detect_draft(file_name: str, first_page_text: str) -> bool:
    if _DRAFT_NAME_RE.search(file_name.lower()):
        return True
    head = [ln for ln in normalize_text(first_page_text).split("\n") if ln.strip()][:3]
    return any(_DRAFT_LINE_RE.match(ln) for ln in head)
