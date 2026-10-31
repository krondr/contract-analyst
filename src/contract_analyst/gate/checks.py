"""Individual grounding and plausibility checks. Each returns ok + a machine-readable code."""

from __future__ import annotations

import re
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Literal

from pydantic import BaseModel
from rapidfuzz import fuzz

from contract_analyst.ingest.normalize import is_redacted, normalize_text
from contract_analyst.models import Page, PageMethod
from contract_analyst.parsing.patterns import DOC_NUMBER_RE, find_dates, find_doc_numbers, find_numbers

ValueKind = Literal["date", "number", "money", "doc_number", "text"]

MIN_DATE, MAX_DATE = date(2000, 1, 1), date(2035, 12, 31)
_TYPOGRAPHY = str.maketrans({"«": '"', "»": '"', "“": '"', "”": '"', "„": '"', "–": "-", "—": "-"})
_TOKEN_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё-]{3,}")


class CheckResult(BaseModel):
    ok: bool
    code: str
    reason: str | None = None


def _norm(text: str) -> str:
    return " ".join(normalize_text(text).translate(_TYPOGRAPHY).casefold().split())


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(_norm(text))


def _ok(code: str) -> CheckResult:
    return CheckResult(ok=True, code=code)


def _fail(code: str, reason: str) -> CheckResult:
    return CheckResult(ok=False, code=code, reason=reason)


def check_quote_present(quote: str | None, page: Page, fuzzy_threshold: int) -> CheckResult:
    if not quote or not quote.strip():
        return _fail("no_quote", "цитата не указана")
    q, text = _norm(quote), _norm(page.text)
    if q in text:
        return _ok("quote_present")
    if page.method == PageMethod.OCR and fuzz.partial_ratio(q, text) >= fuzzy_threshold:
        return _ok("quote_present_fuzzy")
    return _fail("quote_not_found", f"цитата не найдена на странице {page.number}")


def _as_date(value: str) -> date | None:
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _as_decimal(value: str) -> Decimal | None:
    try:
        return Decimal(value.replace(" ", "").replace(",", "."))
    except InvalidOperation:
        return None


def check_value_in_quote(value: str, quote: str, kind: ValueKind) -> CheckResult:
    if kind == "date":
        d = _as_date(value)
        found = [x for x, _ in find_dates(normalize_text(quote))]
        return _ok("value_in_quote") if d and d in found else _fail("value_not_in_quote", f"даты {value} нет в цитате")
    if kind in ("money", "number"):
        n = _as_decimal(value)
        found = find_numbers(normalize_text(quote))
        return _ok("value_in_quote") if n is not None and n in found else _fail(
            "value_not_in_quote", f"числа {value} нет в цитате")
    if kind == "doc_number":
        return _ok("value_in_quote") if value in find_doc_numbers(normalize_text(quote)) else _fail(
            "value_not_in_quote", f"номера {value} нет в цитате")
    tokens = _tokens(value)
    if not tokens:
        return _ok("value_in_quote")
    quoted = _norm(quote)
    share = sum(1 for t in tokens if t in quoted) / len(tokens)
    return _ok("value_in_quote") if share >= 0.6 else _fail("value_not_in_quote", "значение не следует из цитаты")


def check_format(value: str, kind: ValueKind) -> CheckResult:
    if kind == "doc_number":
        return _ok("format") if DOC_NUMBER_RE.fullmatch(value) else _fail("bad_format", f"номер {value} не по формату")
    if kind == "date":
        d = _as_date(value)
        return _ok("format") if d and MIN_DATE <= d <= MAX_DATE else _fail("bad_format", f"дата {value} вне диапазона")
    if kind in ("money", "number"):
        n = _as_decimal(value)
        if n is None or (kind == "money" and n < 0):
            return _fail("bad_format", f"сумма {value} некорректна")
        return _ok("format")
    return _ok("format") if value.strip() else _fail("bad_format", "пустое значение")


def check_ocr_confidence(quote: str, page: Page, threshold: float) -> CheckResult:
    if page.method != PageMethod.OCR:
        return _ok("not_ocr")
    by_text: dict[str, float] = {}
    for word in page.words:
        key = _norm(word.text)
        by_text[key] = max(by_text.get(key, 0.0), word.conf)
    confs = [by_text[t] for t in _norm(quote).split() if t in by_text]
    if not confs:
        return _fail("ocr_low_confidence", "уверенность OCR для цитаты не подтверждена")
    low = min(confs)
    return _ok("ocr_confident") if low >= threshold else _fail(
        "ocr_low_confidence", f"уверенность OCR {low:.2f} ниже порога {threshold:.2f}")


def check_not_redacted(quote: str | None, value: str | None) -> CheckResult:
    if (quote and is_redacted(quote)) or (value and is_redacted(value)):
        return _fail("redacted", "поле вымарано в источнике")
    return _ok("not_redacted")
