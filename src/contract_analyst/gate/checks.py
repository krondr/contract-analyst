"""Individual grounding and plausibility checks. Each returns ok + a machine-readable code."""

from __future__ import annotations

import re
from datetime import date
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel
from rapidfuzz import fuzz

from contract_analyst.ingest.normalize import is_redacted, normalize_text
from contract_analyst.models import Page, PageMethod
from contract_analyst.parsing.patterns import (
    DOC_NUMBER_RE,
    date_spans,
    doc_number_spans,
    find_dates,
    find_doc_numbers,
    find_numbers,
    number_readings,
)

ValueKind = Literal["date", "number", "money", "doc_number", "text"]
Span = tuple[int, int]  # offsets in normalize_text(page.text)
Figure = date | Decimal | str  # what a value states and the page must show: a date, an amount, a document number

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


@lru_cache(maxsize=256)
def _indexed(text: str) -> tuple[str, str, tuple[int, ...]]:
    """normalize_text(text); _norm(text); for every character of the latter, its offset in the former."""
    source = normalize_text(text)
    chars: list[str] = []
    index: list[int] = []
    for i, ch in enumerate(source):
        if ch.isspace():
            if chars and chars[-1] != " ":
                chars.append(" ")
                index.append(i)
            continue
        for folded in ch.translate(_TYPOGRAPHY).casefold():
            chars.append(folded)
            index.append(i)
    if chars and chars[-1] == " ":
        chars.pop()
        index.pop()
    return source, "".join(chars), tuple(index)


def _whole_words(text: str, start: int, end: int) -> Span:
    """A quote that starts inside "21500" is no evidence for 1500: spans grow to whole words."""
    while start > 0 and not text[start - 1].isspace():
        start -= 1
    while end < len(text) and not text[end].isspace():
        end += 1
    return start, end


def locate_quote(quote: str | None, page: Page, fuzzy_threshold: int) -> tuple[CheckResult, list[Span]]:
    """Where the quote is on the page: every verbatim occurrence, or on a scan the fuzzily aligned part."""
    if not quote or not quote.strip():
        return _fail("no_quote", "цитата не указана"), []
    q = _norm(quote)
    source, flat, index = _indexed(page.text)
    spans: list[Span] = []
    at = flat.find(q)
    while at >= 0:
        spans.append(_whole_words(source, index[at], index[at + len(q) - 1] + 1))
        at = flat.find(q, at + 1)
    if spans:
        return _ok("quote_present"), spans
    if page.method == PageMethod.OCR:
        aligned = fuzz.partial_ratio_alignment(q, flat, score_cutoff=fuzzy_threshold)
        if aligned is not None and aligned.dest_end > aligned.dest_start:
            span = _whole_words(source, index[aligned.dest_start], index[aligned.dest_end - 1] + 1)
            return _ok("quote_present_fuzzy"), [span]
    return _fail("quote_not_found", f"цитата не найдена на странице {page.number}"), []


def check_quote_present(quote: str | None, page: Page, fuzzy_threshold: int) -> CheckResult:
    return locate_quote(quote, page, fuzzy_threshold)[0]


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


def _figures(text: str, split_cells: bool = True) -> list[tuple[Figure, int, int, bool]]:
    """Dates, document numbers and amounts with positions and ambiguity; the digits of a date or a document
    number are part of it, not amounts."""
    found: list[tuple[Figure, int, int, bool]] = [(d, a, b, False) for d, a, b in date_spans(text)]
    found += [(n, a, b, False) for n, a, b in doc_number_spans(text)]
    taken = [(a, b) for _, a, b, _ in found]
    found += [(r.value, r.start, r.end, r.ambiguous) for r in number_readings(text, split_cells)
              if not any(a < r.end and r.start < b for a, b in taken)]
    return sorted(found, key=lambda f: f[1])


def _value_figures(value: str, kind: ValueKind) -> list[Figure]:
    if kind == "date":
        return [d] if (d := _as_date(value)) else []
    if kind in ("money", "number"):
        return [n] if (n := _as_decimal(value)) is not None else []
    if kind == "doc_number":
        return [value]
    return [f for f, *_ in _figures(normalize_text(value), split_cells=False)]


def _show(figure: Figure) -> str:
    return figure.strftime("%d.%m.%Y") if isinstance(figure, date) else str(figure)


def check_value_in_quote(value: str, quote: str | None, kind: ValueKind) -> CheckResult:
    if not quote:
        return _fail("no_quote", "цитата не указана")
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
    quoted_figures = {f for f, *_ in _figures(normalize_text(quote))}
    if missing := [f for f in _value_figures(value, "text") if f not in quoted_figures]:
        return _fail("value_not_in_quote", f"{_show(missing[0])} из значения нет в цитате")
    tokens = _tokens(value)
    if not tokens:
        return _ok("value_in_quote")
    quoted = _norm(quote)
    share = sum(1 for t in tokens if t in quoted) / len(tokens)
    return _ok("value_in_quote") if share >= 0.6 else _fail("value_not_in_quote", "значение не следует из цитаты")


def _word_confidences(page: Page) -> list[tuple[int, int, float | None]]:
    """Every word of normalize_text(page.text) with the confidence of the OCR word at the same position."""
    source = normalize_text(page.text)
    words = [(m.start(), m.end()) for m in re.finditer(r"\S+", source)]
    confs: list[float | None] = [None] * len(words)
    matcher = SequenceMatcher(None, [_norm(source[a:b]) for a, b in words], [_norm(w.text) for w in page.words],
                              autojunk=False)
    for block in matcher.get_matching_blocks():
        for k in range(block.size):
            confs[block.a + k] = page.words[block.b + k].conf
    return [(a, b, conf) for (a, b), conf in zip(words, confs)]


def confident(start: int, end: int, words: list[tuple[int, int, float | None]], threshold: float) -> bool:
    """Every OCR word the text at start:end is read from is recognized at or above the threshold."""
    return all(conf is not None and conf >= threshold for a, b, conf in words if a < end and start < b)


def check_value_on_page(value: str, kind: ValueKind, page: Page, spans: list[Span],
                        conf_threshold: float) -> CheckResult:
    """The value is read from the page, not from the model's quote: every figure it states must occur in a
    located quote span, on a scan in confidently recognized words, and in a reading that is not ambiguous."""
    source = normalize_text(page.text)
    on_page = [f for f in _figures(source) if any(a <= f[1] and f[2] <= b for a, b in spans)]
    words = _word_confidences(page) if page.method == PageMethod.OCR else None
    ambiguous: Figure | None = None
    for figure in _value_figures(value, kind):
        seen = [(a, b, amb) for f, a, b, amb in on_page if f == figure]
        if not seen:
            return _fail("value_not_on_page", f"{_show(figure)} нет на странице {page.number} в цитируемом месте")
        if words is not None:
            seen = [(a, b, amb) for a, b, amb in seen if confident(a, b, words, conf_threshold)]
            if not seen:
                return _fail("ocr_value_unconfirmed", f"OCR не уверен в «{_show(figure)}» на странице {page.number}")
        if all(amb for *_, amb in seen):
            ambiguous = figure
    if ambiguous is not None:
        return _fail("ambiguous_number", f"«{_show(ambiguous)}» может быть частью числа или соседней ячейкой таблицы")
    return _ok("value_on_page")


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
