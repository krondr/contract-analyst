"""The validation gate: a value reaches the knowledge base only if every check passes."""

from __future__ import annotations

from datetime import date
from functools import lru_cache

from pydantic import BaseModel
from rapidfuzz import fuzz

from contract_analyst.config import Settings
from contract_analyst.gate.checks import (
    CheckResult,
    ValueKind,
    _norm,
    check_format,
    check_not_redacted,
    check_ocr_confidence,
    check_quote_present,
    check_value_in_quote,
)
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
from contract_analyst.parsing.patterns import find_dates, parse_money
from contract_analyst.understand.schemas import OpOut, RateOut


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


_CLAUSE_FUZZY = 90
_norm_cached = lru_cache(maxsize=8192)(_norm)


def clause_at(doc: DocumentText, page: int, quote: str) -> str | None:
    """Reference of the clause holding the quote: the nearest clause that starts on or before the quote's page."""
    q = _norm(quote)
    before = [c for c in doc.clauses if c.page <= page]
    if not q or not before:
        return None
    for clause in reversed(before):
        if q in _norm_cached(clause.text):
            return clause.ref
    score, best = max(((fuzz.partial_ratio(q, _norm_cached(c.text)), c) for c in before if c.page >= page - 1),
                      key=lambda item: item[0], default=(0, None))
    return best.ref if best is not None and score >= _CLAUSE_FUZZY else None


def _source(doc: DocumentText, page: int | None, quote: str | None) -> Source | None:
    if not page or not quote:
        return None
    return Source(doc_id=doc.doc_id, page=page, quote=quote, clause_id=clause_at(doc, page, quote))


def _reason(check: CheckResult) -> str:
    return f"{check.code}: {check.reason}"


def _confidence(page: Page) -> float:
    return page.quality.ocr_mean_conf if page.method == PageMethod.OCR and page.quality.ocr_mean_conf else 1.0


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
    for check in (check_quote_present(c.quote, page, settings.ocr_quote_fuzzy_threshold),
                  check_value_in_quote(c.value, c.quote, c.kind),
                  check_format(c.value, c.kind),
                  check_ocr_confidence(c.quote, page, settings.ocr_word_conf_threshold)):
        if not check.ok:
            return Extracted[str](value=c.value, status=ValueStatus.REJECTED, source=_source(doc, c.page, c.quote),
                                  reason=_reason(check))
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


def _reanchor(row: RateOut, amount, doc: DocumentText) -> tuple[int, str] | None:
    """Find the evidence for a table rate: a line with every label word and the amount, or — when cells are
    extracted one per line — the label followed directly by its numbers in the text stream."""
    import re

    from contract_analyst.ingest.normalize import normalize_text
    from contract_analyst.parsing.patterns import find_numbers

    def words(text: str) -> set[str]:
        return set(re.findall(r"[0-9a-zа-яё]+", normalize_text(text).casefold()))

    label = words(row.label)
    label_flat = " ".join(normalize_text(row.label).split()).casefold()
    pages = [doc.pages[row.page - 1]] if 1 <= row.page <= len(doc.pages) else []
    pages += [p for p in doc.pages if p.number != row.page]
    for page in pages:
        for line in page.text.split("\n"):
            if amount in find_numbers(line) and label and label <= words(line):
                return page.number, line.strip()
        flat = " ".join(normalize_text(page.text).split())
        for m in re.finditer(re.escape(label_flat), flat.casefold()) if label_flat else []:
            end = m.end()
            for token in re.finditer(r"\S+", flat[m.end():m.end() + 80]):
                text = token.group(0)
                if re.fullmatch(r"[:–—-]", text):
                    continue
                if not re.fullmatch(r"\d[\d.,]*", text):
                    break
                end = m.end() + token.end()
                if amount in find_numbers(text):
                    return page.number, flat[m.start():end]
    return None


def gate_rate(row: RateOut, doc: DocumentText, settings: Settings) -> RateRow:
    amount = parse_money(row.amount)
    reanchored = False
    if amount is not None and not check_value_in_quote(str(amount), row.quote, "money").ok:
        found = _reanchor(row, amount, doc)
        if found:
            row = row.model_copy(update={"page": found[0], "quote": found[1]})
            reanchored = True
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
    checks = [check_not_redacted(row.quote, row.amount), check_quote_present(row.quote, page, settings.ocr_quote_fuzzy_threshold),
              check_value_in_quote(str(amount), row.quote, "money"), check_format(str(amount), "money"),
              check_ocr_confidence(row.quote, page, settings.ocr_word_conf_threshold)]
    dates_on_doc = _doc_dates(doc)
    for d in (base["valid_from"], base["valid_to"]):
        if d and d not in dates_on_doc:
            checks.append(CheckResult(ok=False, code="date_not_in_document", reason=f"даты {d} нет в документе"))
    for check in checks:
        if not check.ok:
            status = ValueStatus.REDACTED if check.code == "redacted" else ValueStatus.REJECTED
            return RateRow(**base, status=status, reason=_reason(check))
    return RateRow(**base, status=ValueStatus.ACCEPTED, reason="quote_reanchored" if reanchored else None)


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
    checks = [check_quote_present(op.quote, page, settings.ocr_quote_fuzzy_threshold),
              check_ocr_confidence(op.quote, page, settings.ocr_word_conf_threshold)]
    if op.op == OpType.INVALIDATE_DOCUMENT:
        checks.append(check_value_in_quote(op.target or "", op.quote, "doc_number"))
    dates_on_doc = _doc_dates(doc)
    for d in (base["effective_from"], base["effective_to"]):
        if d and d not in dates_on_doc:
            checks.append(CheckResult(ok=False, code="date_not_in_document", reason=f"даты {d} нет в документе"))
    for check in checks:
        if not check.ok:
            return AmendmentOp(**base, status=ValueStatus.REJECTED, reason=_reason(check))
    return AmendmentOp(**base, status=ValueStatus.ACCEPTED)
