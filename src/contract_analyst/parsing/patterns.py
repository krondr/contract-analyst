"""Deterministic patterns for Russian contract texts: document numbers, dates, amounts."""

from __future__ import annotations

import re
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import NamedTuple

DOC_NUMBER_RE = re.compile(r"(?<![0-9A-Za-z])D\d{9}(?:-\d{2})?(?!\d)")

RU_MONTHS = {
    "январ": 1, "феврал": 2, "март": 3, "апрел": 4, "ма": 5, "июн": 6,
    "июл": 7, "август": 8, "сентябр": 9, "октябр": 10, "ноябр": 11, "декабр": 12,
}
_MONTH_ALT = r"январ[яь]|феврал[яь]|марта?|апрел[яь]|ма[яй]|июн[яь]|июл[яь]|августа?|сентябр[яь]|октябр[яь]|ноябр[яь]|декабр[яь]"
_TEXT_DATE_RE = re.compile(rf"«?\s*(\d{{1,2}})\s*»?\s+({_MONTH_ALT})\s+(\d{{4}})", re.IGNORECASE)
_NUM_DATE_RE = re.compile(r"(?<!\d)(\d{1,2})\.(\d{1,2})\.(\d{4})(?!\d)")
_MONEY_RE = re.compile(r"(?<![\d.,])\d{1,3}(?:[  ]\d{3})+(?!\d)(?:[.,]\d+)?|(?<![\d.,])\d+(?:[.,]\d+)?")
_CELLS_ONLY = re.compile(r"[\s|]*(?:\d+(?:[.,]\d+)?[\s|]*)*")
# What follows a number in a sentence and never starts the next table row: a lowercase word, a bracketed
# amount in words, a unit.
_SENTENCE_GOES_ON = re.compile(r"\s*(?:[a-zа-яё(%₽]|(?i:руб|ндс|коп))")


class NumberReading(NamedTuple):
    value: Decimal
    start: int
    end: int
    ambiguous: bool  # "270 500" on a table row: one amount, or two cells


def _month_number(word: str) -> int | None:
    word = word.lower()
    for prefix, number in sorted(RU_MONTHS.items(), key=lambda kv: -len(kv[0])):
        if word.startswith(prefix):
            return number
    return None


def _safe_date(year: int, month: int | None, day: int) -> date | None:
    if month is None:
        return None
    try:
        return date(year, month, day)
    except ValueError:
        return None


def date_spans(text: str) -> list[tuple[date, int, int]]:
    """Every date with its position, in text order."""
    found: list[tuple[date, int, int]] = []
    for m in _TEXT_DATE_RE.finditer(text):
        d = _safe_date(int(m.group(3)), _month_number(m.group(2)), int(m.group(1)))
        if d:
            found.append((d, m.end() - len(m.group(0).lstrip()), m.end()))
    for m in _NUM_DATE_RE.finditer(text):
        d = _safe_date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        if d:
            found.append((d, m.start(), m.end()))
    return sorted(found, key=lambda item: item[1])


def find_dates(text: str) -> list[tuple[date, str]]:
    return [(d, text[start:end]) for d, start, end in date_spans(text)]


def parse_ru_date(s: str) -> date | None:
    found = find_dates(s)
    return found[0][0] if found else None


# OCR on scans reads the leading Latin D as 0, Cyrillic О or Д; the -NN suffix keeps this from matching
# arbitrary ten-digit numbers.
_OCR_NUMBER_RE = re.compile(r"(?<![0-9A-Za-zА-Яа-я])(?:D|[0ОOДD](?=\d{9}-\d{2}(?!\d)))(\d{9}(?:-\d{2})?)(?!\d)")


def doc_number_spans(text: str) -> list[tuple[str, int, int]]:
    return [(f"D{m.group(1)}", m.start(), m.end()) for m in _OCR_NUMBER_RE.finditer(text)]


def find_doc_numbers(text: str) -> list[str]:
    return [number for number, _, _ in doc_number_spans(text)]


def _to_decimal(raw: str) -> Decimal | None:
    try:
        return Decimal(raw.replace(" ", "").replace(" ", "").replace(",", "."))
    except InvalidOperation:
        return None


def ends_table_row(text: str, end: int) -> bool:
    """Only numeric cells follow position `end` on its line, and the next line does not continue a sentence."""
    line_end = text.find("\n", end)
    line_end = len(text) if line_end < 0 else line_end
    return bool(_CELLS_ONLY.fullmatch(text, end, line_end)) and not _SENTENCE_GOES_ON.match(text, line_end)


def _may_be_cells(text: str, start: int, end: int) -> bool:
    """A space-grouped number may be adjacent table cells that PyMuPDF or OCR put on one line: its row has no
    explicit cell separator."""
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    return "|" not in text[line_start:None if line_end < 0 else line_end] and ends_table_row(text, end)


def number_readings(text: str, split_cells: bool = True) -> list[NumberReading]:
    """Amount-like numbers in order, with positions; numeric dates are not counted.

    "1 500 рублей" is one amount. "Базовый 280 520" on a table row may be one amount or two cells: then the
    grouped reading and each cell are all kept, marked ambiguous, the grouped one first.
    """
    masked = _NUM_DATE_RE.sub(lambda m: " " * len(m.group(0)), text)
    found: list[NumberReading] = []
    for m in _MONEY_RE.finditer(masked):
        value = _to_decimal(m.group(0))
        if value is None:
            continue
        cells = split_cells and bool(re.search(r"\s", m.group(0))) and _may_be_cells(masked, m.start(), m.end())
        found.append(NumberReading(value, m.start(), m.end(), cells))
        if cells:
            for part in re.finditer(r"\S+", m.group(0)):
                if (n := _to_decimal(part.group(0))) is not None:
                    found.append(NumberReading(n, m.start() + part.start(), m.start() + part.end(), True))
    return found


def find_numbers(text: str) -> list[Decimal]:
    return [r.value for r in number_readings(text)]


def parse_money(s: str) -> Decimal | None:
    numbers = find_numbers(s)
    return numbers[0] if numbers else None
