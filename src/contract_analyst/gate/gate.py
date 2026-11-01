"""The validation gate: a value reaches the knowledge base only if every check passes."""

from __future__ import annotations

import re
from collections.abc import Iterator
from datetime import date
from decimal import Decimal

from pydantic import BaseModel

from contract_analyst.config import Settings
from contract_analyst.gate.checks import (
    CheckResult,
    ValueKind,
    check_format,
    check_not_redacted,
    check_ocr_confidence,
    check_value_in_quote,
    check_value_on_page,
    locate_quote,
)
from contract_analyst.ingest.normalize import normalize_text
from contract_analyst.models import (
    AmendmentOp,
    DocumentText,
    Extracted,
    OpType,
    Page,
    PageMethod,
    RateRow,
    Source,
    ValueStatus,
)
from contract_analyst.parsing.patterns import find_dates, find_numbers, parse_money
from contract_analyst.understand.schemas import OpOut, RateOut

# The evidence fits more than one reading: doubtful rather than wrong.
_CONFLICT_CODES = {"ambiguous_number", "reanchor_ambiguous"}


class Candidate(BaseModel):
    field: str
    value: str | None
    kind: ValueKind
    page: int | None
    quote: str | None
    redacted: bool = False


def _page(doc: DocumentText, number: int | None) -> Page | None:
    if number is None or not 1 <= number <= len(doc.pages):
        return None
    return doc.pages[number - 1]


def _source(doc: DocumentText, page: int | None, quote: str | None) -> Source | None:
    return Source(doc_id=doc.doc_id, page=page, quote=quote) if page and quote else None


def _reason(check: CheckResult) -> str:
    return f"{check.code}: {check.reason}"


def _confidence(page: Page) -> float:
    return page.quality.ocr_mean_conf if page.method == PageMethod.OCR and page.quality.ocr_mean_conf else 1.0


def _status(check: CheckResult) -> ValueStatus:
    if check.code == "redacted":
        return ValueStatus.REDACTED
    return ValueStatus.CONFLICT if check.code in _CONFLICT_CODES else ValueStatus.REJECTED


def _first_failure(checks: Iterator[CheckResult]) -> CheckResult | None:
    """Checks run lazily in order: a later check may rely on an earlier one having passed."""
    return next((check for check in checks if not check.ok), None)


def apply_gate(c: Candidate, doc: DocumentText, settings: Settings) -> Extracted[str]:
    if c.redacted:
        return Extracted[str](status=ValueStatus.REDACTED, source=_source(doc, c.page, c.quote), reason="redacted")
    if c.value is None:
        return Extracted[str](status=ValueStatus.ABSENT, reason="absent")
    redaction = check_not_redacted(c.quote, c.value)
    if not redaction.ok:
        return Extracted[str](status=ValueStatus.REDACTED, source=_source(doc, c.page, c.quote), reason=_reason(redaction))
    page = _page(doc, c.page)
    if page is None:
        return Extracted[str](value=c.value, status=ValueStatus.REJECTED, reason=f"bad_page: страница {c.page} не существует")
    threshold = settings.ocr_word_conf_threshold

    def checks() -> Iterator[CheckResult]:
        found, spans = locate_quote(c.quote, page, settings.ocr_quote_fuzzy_threshold)
        yield found
        yield check_value_in_quote(c.value, c.quote, c.kind)
        yield check_format(c.value, c.kind)
        yield check_ocr_confidence(c.quote, page, threshold)
        yield check_value_on_page(c.value, c.kind, page, spans, threshold)

    if failed := _first_failure(checks()):
        return Extracted[str](value=c.value, status=_status(failed), source=_source(doc, c.page, c.quote),
                              reason=_reason(failed))
    return Extracted[str](value=c.value, status=ValueStatus.ACCEPTED, source=_source(doc, c.page, c.quote),
                          confidence=_confidence(page))


def _iso(value: str | None) -> date | None | str:
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return "bad"


def _doc_dates(doc: DocumentText) -> set[date]:
    return {d for p in doc.pages for d, _ in find_dates(p.text)}


def _words(text: str) -> set[str]:
    return set(re.findall(r"[0-9a-zа-яё]+", normalize_text(text).casefold()))


def _reanchor(row: RateOut, amount: Decimal, doc: DocumentText) -> list[str]:
    """Evidence for a table rate on its own page, one quote per place: a line with every label word and the
    amount, or — when cells are extracted one per line — the label followed directly by its numbers."""
    page = _page(doc, row.page)
    label = _words(row.label)
    if page is None or not label:
        return []
    text = normalize_text(page.text)
    found: list[tuple[int, int, str]] = []
    start = 0
    for line in text.split("\n"):
        if label <= _words(line) and amount in find_numbers(line):
            found.append((start, start + len(line), line.strip()))
        start += len(line) + 1
    flat = text.replace("\n", " ")  # same length: offsets stay valid
    for m in re.finditer(re.escape(" ".join(normalize_text(row.label).split())), flat, re.IGNORECASE):
        for token in re.finditer(r"\S+", flat[m.end():m.end() + 80]):
            if re.fullmatch(r"[:–—-]", token.group(0)):
                continue
            if not re.fullmatch(r"\d[\d.,]*", token.group(0)):
                break
            if amount in find_numbers(token.group(0)):
                end = m.end() + token.end()
                found.append((m.start(), end, flat[m.start():end]))
                break
    places: list[list] = []
    for start, end, quote in sorted(found, key=lambda f: (f[0], -f[1])):  # the whole line before its label match
        if places and start < places[-1][1]:
            places[-1][1] = max(places[-1][1], end)
        else:
            places.append([start, end, quote])
    return [quote for _, _, quote in places]


def gate_rate(row: RateOut, doc: DocumentText, settings: Settings) -> RateRow:
    amount = parse_money(row.amount)
    places: list[str] = []
    if amount is not None and not check_value_in_quote(str(amount), row.quote, "money").ok:
        places = _reanchor(row, amount, doc)
        if len(places) == 1:
            row = row.model_copy(update={"quote": places[0]})
    valid_from, valid_to = _iso(row.valid_from), _iso(row.valid_to)
    base = {"label": row.label, "amount": amount if amount is not None else 0, "vat": row.vat,
                "valid_from": valid_from if isinstance(valid_from, date) else None,
                "valid_to": valid_to if isinstance(valid_to, date) else None, "source": _source(doc, row.page, row.quote)}
    if amount is None:
        return RateRow(**base, status=ValueStatus.REJECTED, reason="bad_format: сумма не распознана")
    if "bad" in (valid_from, valid_to):
        return RateRow(**base, status=ValueStatus.REJECTED, reason="bad_format: дата периода не распознана")
    page = _page(doc, row.page)
    if page is None:
        return RateRow(**base, status=ValueStatus.REJECTED, reason=f"bad_page: страница {row.page} не существует")
    threshold = settings.ocr_word_conf_threshold

    def checks() -> Iterator[CheckResult]:
        yield check_not_redacted(row.quote, row.amount)
        if len(places) > 1:
            yield CheckResult(ok=False, code="reanchor_ambiguous", reason=(
                f"сумма {amount} со словами «{row.label}» есть на странице {row.page} в {len(places)} местах"))
        found, spans = locate_quote(row.quote, page, settings.ocr_quote_fuzzy_threshold)
        yield found
        yield check_value_in_quote(str(amount), row.quote, "money")
        yield check_format(str(amount), "money")
        yield check_ocr_confidence(row.quote, page, threshold)
        yield check_value_on_page(str(amount), "money", page, spans, threshold)
        dates_on_doc = _doc_dates(doc)
        for d in (base["valid_from"], base["valid_to"]):
            if d and d not in dates_on_doc:
                yield CheckResult(ok=False, code="date_not_in_document", reason=f"даты {d} нет в документе")

    if failed := _first_failure(checks()):
        return RateRow(**base, status=_status(failed), reason=_reason(failed))
    return RateRow(**base, status=ValueStatus.ACCEPTED, reason="quote_reanchored" if places else None)


def gate_op(op: OpOut, doc: DocumentText, settings: Settings) -> AmendmentOp:
    eff_from, eff_to = _iso(op.effective_from), _iso(op.effective_to)
    base = {"op": op.op, "target": op.target, "new_text": op.new_text_excerpt,
                "effective_from": eff_from if isinstance(eff_from, date) else None,
                "effective_to": eff_to if isinstance(eff_to, date) else None, "source": _source(doc, op.page, op.quote)}
    if "bad" in (eff_from, eff_to):
        return AmendmentOp(**base, status=ValueStatus.REJECTED, reason="bad_format: дата не распознана")
    page = _page(doc, op.page)
    if page is None:
        return AmendmentOp(**base, status=ValueStatus.REJECTED, reason=f"bad_page: страница {op.page} не существует")
    threshold = settings.ocr_word_conf_threshold

    def checks() -> Iterator[CheckResult]:
        found, spans = locate_quote(op.quote, page, settings.ocr_quote_fuzzy_threshold)
        yield found
        yield check_ocr_confidence(op.quote, page, threshold)
        if op.op == OpType.INVALIDATE_DOCUMENT:
            yield check_value_in_quote(op.target or "", op.quote, "doc_number")
            yield check_value_on_page(op.target or "", "doc_number", page, spans, threshold)
        dates_on_doc = _doc_dates(doc)
        for d in (base["effective_from"], base["effective_to"]):
            if d and d not in dates_on_doc:
                yield CheckResult(ok=False, code="date_not_in_document", reason=f"даты {d} нет в документе")

    if failed := _first_failure(checks()):
        return AmendmentOp(**base, status=_status(failed), reason=_reason(failed))
    return AmendmentOp(**base, status=ValueStatus.ACCEPTED)
