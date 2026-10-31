"""Deterministic patterns for Russian contract texts: document numbers, dates, amounts."""

from __future__ import annotations

import re
from datetime import date
from decimal import Decimal, InvalidOperation

DOC_NUMBER_RE = re.compile(r"(?<![0-9A-Za-z])D\d{9}(?:-\d{2})?(?!\d)")

RU_MONTHS = {
    "январ": 1, "феврал": 2, "март": 3, "апрел": 4, "ма": 5, "июн": 6,
    "июл": 7, "август": 8, "сентябр": 9, "октябр": 10, "ноябр": 11, "декабр": 12,
}
_MONTH_ALT = r"январ[яь]|феврал[яь]|марта?|апрел[яь]|ма[яй]|июн[яь]|июл[яь]|августа?|сентябр[яь]|октябр[яь]|ноябр[яь]|декабр[яь]"
_TEXT_DATE_RE = re.compile(rf"«?\s*(\d{{1,2}})\s*»?\s+({_MONTH_ALT})\s+(\d{{4}})", re.IGNORECASE)
_NUM_DATE_RE = re.compile(r"(?<!\d)(\d{1,2})\.(\d{1,2})\.(\d{4})(?!\d)")
_MONEY_RE = re.compile(r"(?<![\d.,])\d{1,3}(?:[  ]\d{3})+(?!\d)(?:[.,]\d+)?|(?<![\d.,])\d+(?:[.,]\d+)?")


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


def find_dates(text: str) -> list[tuple[date, str]]:
    found: list[tuple[int, date, str]] = []
    for m in _TEXT_DATE_RE.finditer(text):
        d = _safe_date(int(m.group(3)), _month_number(m.group(2)), int(m.group(1)))
        if d:
            found.append((m.start(), d, m.group(0).strip()))
    for m in _NUM_DATE_RE.finditer(text):
        d = _safe_date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        if d:
            found.append((m.start(), d, m.group(0)))
    return [(d, span) for _, d, span in sorted(found, key=lambda item: item[0])]


def parse_ru_date(s: str) -> date | None:
    found = find_dates(s)
    return found[0][0] if found else None


# OCR on scans reads the leading Latin D as 0, Cyrillic О or Д; the -NN suffix keeps this from matching
# arbitrary ten-digit numbers.
_OCR_NUMBER_RE = re.compile(r"(?<![0-9A-Za-zА-Яа-я])(?:D|[0ОOДD](?=\d{9}-\d{2}(?!\d)))(\d{9}(?:-\d{2})?)(?!\d)")


def find_doc_numbers(text: str) -> list[str]:
    return [f"D{m.group(1)}" for m in _OCR_NUMBER_RE.finditer(text)]


def _to_decimal(raw: str) -> Decimal | None:
    try:
        return Decimal(raw.replace(" ", "").replace(" ", "").replace(",", "."))
    except InvalidOperation:
        return None


def find_numbers(text: str) -> list[Decimal]:
    """All amount-like numbers in order; dates are removed first so their parts are not counted.

    "270 500" is ambiguous: one amount with a thousands separator or two table cells. Both readings are kept,
    the grouped one first, so parse_money("1 500") is still 1500.
    """
    without_dates = _NUM_DATE_RE.sub(" ", text)
    found: list[Decimal] = []
    for m in _MONEY_RE.finditer(without_dates):
        raw = m.group(0)
        readings = [raw] + (raw.replace("\u00a0", " ").split(" ") if re.search(r"[ \u00a0]", raw) else [])
        found.extend(n for n in (_to_decimal(r) for r in readings) if n is not None)
    return found


def parse_money(s: str) -> Decimal | None:
    numbers = find_numbers(s)
    return numbers[0] if numbers else None
