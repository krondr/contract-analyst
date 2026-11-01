"""LLM understanding by field groups: every value is grounded by the gate before it is kept."""

from __future__ import annotations

from datetime import date

import httpx

from contract_analyst.config import Settings
from contract_analyst.gate.checks import check_quote_present
from contract_analyst.gate.gate import Candidate, apply_gate, gate_op, gate_rate
from contract_analyst.llm.client import LLMClient, Message
from contract_analyst.llm.prompts import load_prompt
from contract_analyst.llm.structured import StructuredFailure, call_structured
from contract_analyst.models import (
    AmendmentOp,
    Clause,
    DocKind,
    DocumentRecord,
    DocumentText,
    Extracted,
    Issue,
    OpType,
    ParentRef,
    Party,
    RateRow,
    ValueStatus,
)
from contract_analyst.parsing.header import HeaderGuess, detect_draft
from contract_analyst.understand.routing import select_clauses
from contract_analyst.understand.schemas import HeaderOut, OpsOut, RatesOut, TermsOut, ValueOut

GROUP_PROMPTS = {"header": "header", "core_terms": "terms", "dealer": "terms", "money": "rates", "ops": "ops"}
GROUP_SCHEMAS = {"header": HeaderOut, "core_terms": TermsOut, "dealer": TermsOut, "money": RatesOut, "ops": OpsOut}
_REGEX_KIND = {"contract": DocKind.MAIN, "supplement": DocKind.SUPPLEMENT, "agreement": DocKind.FRAMEWORK,
               "unknown": DocKind.UNKNOWN}
_HEADER_KINDS = {"number": "doc_number", "doc_date": "date", "city": "text", "parent_number": "doc_number",
                 "parent_date": "date", "effective_from": "date", "effective_to": "date"}
_TIMEOUTS = (TimeoutError, httpx.TimeoutException)


def prompt_versions() -> dict[str, int]:
    return {name: load_prompt(name).version for name in sorted(set(GROUP_PROMPTS.values()))}


def _absent(reason: str = "absent") -> Extracted[str]:
    return Extracted[str](status=ValueStatus.ABSENT, reason=reason)


def _resolve_page(doc: DocumentText, quote: str | None, hint: int | None, settings: Settings) -> int | None:
    if not quote or not doc.pages:
        return hint
    order = ([hint] if hint and 1 <= hint <= len(doc.pages) else []) + [p.number for p in doc.pages if p.number != hint]
    for number in order:
        if check_quote_present(quote, doc.pages[number - 1], settings.ocr_quote_fuzzy_threshold).ok:
            return number
    return hint


def _gate_value(field: str, out: ValueOut, kind: str, doc: DocumentText, settings: Settings) -> Extracted[str]:
    page = _resolve_page(doc, out.quote, out.page, settings)
    return apply_gate(Candidate(field=field, value=out.value, kind=kind, page=page, quote=out.quote,
                                redacted=out.redacted), doc, settings)


def _merge(regex: Extracted[str] | None, llm: Extracted[str] | None) -> Extracted[str]:
    if regex and regex.status == ValueStatus.ACCEPTED:
        if llm and llm.status == ValueStatus.ACCEPTED and llm.value != regex.value:
            return regex.model_copy(update={
                "status": ValueStatus.CONFLICT, "alternatives": [s for s in [llm.source] if s],
                "reason": f"conflict: в тексте найдено {regex.value}, модель указала {llm.value}"})
        return regex
    return llm or regex or _absent()


def _typed_date(ext: Extracted[str]) -> dict:
    data = ext.model_dump()
    try:
        data["value"] = date.fromisoformat(ext.value) if ext.value else None
    except ValueError:
        data["value"] = None
        data["reason"] = f"{ext.reason or ''} (значение: {ext.value})".strip()
    return data


def _call(client: LLMClient, messages: list[Message], schema, issues: list[Issue], doc: DocumentText, group: str):
    try:
        result = call_structured(client, messages, schema)
    except _TIMEOUTS as exc:
        issues.append(Issue(code="llm_timeout", message=f"{doc.file.name}: группа {group} — LLM не ответила: {exc}"))
        return None
    if isinstance(result, StructuredFailure):
        issues.append(Issue(code="llm_truncated", message=f"{doc.file.name}: группа {group} — ответ оборвался на "
                                                          "лимите токенов и после повтора с просьбой ответить короче")
                      if result.truncated else
                      Issue(code="llm_invalid_json",
                            message=f"{doc.file.name}: группа {group} — ответ не прошёл схему после повторов"))
        return None
    return result


def _messages(doc: DocumentText, group: str, batch: list[Clause], settings: Settings) -> list[Message]:
    prompt = load_prompt(GROUP_PROMPTS[group])
    system = prompt.render(
        fields=[(f, settings.field_descriptions.get(f, f)) for f in settings.field_groups[group]],
        categories=settings.subject_categories, kinds=[k.value for k in DocKind], ops=[o.value for o in OpType])
    body = "\n\n".join(f"[стр. {c.page}] {c.text}" for c in batch)
    return [{"role": "system", "content": system}, {"role": "user", "content": f"Документ: {doc.file.name}\n\n{body}"}]


def _regex_candidates(header: HeaderGuess, doc: DocumentText, settings: Settings) -> dict[str, Extracted[str] | None]:
    values = {"number": header.number, "doc_date": header.doc_date, "city": header.city,
              "parent_number": header.parent_number, "parent_date": header.parent_date}
    spans = {"number": "number", "doc_date": "doc_date", "city": "city", "parent_number": "parent", "parent_date": "parent"}
    found: dict[str, Extracted[str] | None] = {}
    for field, value in values.items():
        if value is None:
            found[field] = None
            continue
        text = value.isoformat() if isinstance(value, date) else str(value)
        quote = header.spans.get(spans[field])
        found[field] = apply_gate(Candidate(field=field, value=text, kind=_HEADER_KINDS[field], page=1, quote=quote),
                                  doc, settings)
    return found



class _Acc:
    """Accumulates one document's values across groups and batches."""

    def __init__(self) -> None:
        self.header: HeaderOut | None = None
        self.header_values: dict[str, Extracted[str]] = {}
        self.parties: list[Party] = []
        self.terms: dict[str, Extracted[str]] = {}
        self.rates: list[RateRow] = []
        self.ops: list[AmendmentOp] = []
        self.ops_effective: Extracted[str] | None = None
        self.issues: list[Issue] = []

    def add_terms(self, field: str, ext: Extracted[str]) -> None:
        old = self.terms.get(field)
        if old is None or old.status != ValueStatus.ACCEPTED:
            self.terms[field] = ext
        elif ext.status == ValueStatus.ACCEPTED and ext.value and ext.value not in (old.value or ""):
            self.terms[field] = old.model_copy(update={"value": f"{old.value}; {ext.value}"})


def _run_group(acc: _Acc, doc: DocumentText, group: str, client: LLMClient, settings: Settings) -> None:
    schema = GROUP_SCHEMAS[group]
    for batch in select_clauses(doc, group, settings):
        out = _call(client, _messages(doc, group, batch, settings), schema, acc.issues, doc, group)
        if out is None:
            continue
        if group == "header":
            acc.header = acc.header or out
            for field, kind in _HEADER_KINDS.items():
                acc.header_values.setdefault(field, _gate_value(field, getattr(out, field), kind, doc, settings))
            acc.parties = acc.parties or [
                Party(role=p.role, org_type=_gate_value("org_type", p.org_type, "text", doc, settings),
                      name=_gate_value("name", p.name, "text", doc, settings)) for p in out.parties]
        elif group in ("core_terms", "dealer"):
            for field in settings.field_groups[group]:
                if field in out.values:
                    acc.add_terms(field, _gate_value(field, out.values[field], "text", doc, settings))
        elif group == "money":
            for row in out.rates:
                page = _resolve_page(doc, row.quote, row.page, settings)
                gated = gate_rate(row.model_copy(update={"page": page or row.page}), doc, settings)
                key = (gated.label, gated.amount, gated.valid_from, gated.valid_to)
                if all((r.label, r.amount, r.valid_from, r.valid_to) != key for r in acc.rates):
                    acc.rates.append(gated)
        elif group == "ops":
            for op in out.ops:
                page = _resolve_page(doc, op.quote, op.page, settings)
                gated = gate_op(op.model_copy(update={"page": page or op.page}), doc, settings)
                if all((o.op, o.target, o.source) != (gated.op, gated.target, gated.source) for o in acc.ops):
                    acc.ops.append(gated)
            if out.effective_from.value and acc.ops_effective is None:
                acc.ops_effective = _gate_value("effective_from", out.effective_from, "date", doc, settings)


def _parent(number: Extracted[str], when: Extracted[str]) -> dict:
    if number.status not in (ValueStatus.ACCEPTED, ValueStatus.CONFLICT):
        return Extracted[ParentRef](status=number.status, reason=number.reason, source=number.source).model_dump()
    stated = _typed_date(when)["value"] if when.status in (ValueStatus.ACCEPTED, ValueStatus.CONFLICT) else None
    conflict = ValueStatus.CONFLICT in (number.status, when.status)
    return Extracted[ParentRef](
        value=ParentRef(number=number.value, date_stated=stated),
        status=ValueStatus.CONFLICT if conflict else ValueStatus.ACCEPTED, source=number.source,
        alternatives=number.alternatives + when.alternatives,
        reason=when.reason if when.status == ValueStatus.CONFLICT else number.reason).model_dump()


def understand_document(doc: DocumentText, header: HeaderGuess, client: LLMClient, settings: Settings,
                        groups: list[str] | None = None) -> DocumentRecord:
    wanted = [g for g in settings.field_groups if groups is None or g in groups]
    acc = _Acc()
    for group in wanted:
        if group == "ops" and acc.header and acc.header.kind == DocKind.MAIN and not header.parent_number:
            continue  # a main contract has nothing to amend; saves LLM calls on long contracts
        _run_group(acc, doc, group, client, settings)

    regex = _regex_candidates(header, doc, settings) if doc.pages else {}
    values = {f: _merge(regex.get(f), acc.header_values.get(f)) for f in _HEADER_KINDS}
    if values["effective_from"].status != ValueStatus.ACCEPTED and acc.ops_effective:
        values["effective_from"] = acc.ops_effective

    kind_value = acc.header.kind if acc.header else _REGEX_KIND[header.title_kind]
    if kind_value == DocKind.FRAMEWORK and values["parent_number"].status == ValueStatus.ACCEPTED:
        kind_value = DocKind.SUPPLEMENT
    kind = Extracted[DocKind](value=kind_value, status=ValueStatus.ACCEPTED,
                              reason="classified_by_llm" if acc.header else "classified_by_title")
    issues = list(acc.issues)
    if header.title_kind == "contract" and (kind_value == DocKind.SUPPLEMENT or header.parent_number
                                            or values["parent_number"].status == ValueStatus.ACCEPTED):
        issues.append(Issue(code="title_body_mismatch",
                            message=f"{doc.file.name}: заголовок «Договор», но по тексту это соглашение к другому документу",
                            sources=[s for s in [values["parent_number"].source] if s]))
    is_draft = detect_draft(doc.file.name, doc.pages[0].text if doc.pages else "")
    if is_draft:
        issues.append(Issue(code="draft", message=f"{doc.file.name}: похоже на проект, подписание не подтверждено"))
    for field in ("number", "doc_date", "parent_number", "parent_date"):
        if values[field].status == ValueStatus.CONFLICT:
            issues.append(Issue(code=f"{field}_conflict", message=values[field].reason or "",
                                sources=[s for s in [values[field].source, *values[field].alternatives] if s]))

    category = acc.header.subject_category if acc.header else "other"
    if category not in settings.subject_categories:
        category = "other"
    title = (acc.header.title if acc.header and acc.header.title else header.spans.get("number")) or doc.file.name
    return DocumentRecord(
        doc_id=doc.doc_id, file_name=doc.file.name, title=title, is_draft=is_draft,
        kind=kind, number=values["number"].model_dump(), doc_date=_typed_date(values["doc_date"]),
        city=values["city"].model_dump(), parent=_parent(values["parent_number"], values["parent_date"]),
        effective_from=_typed_date(values["effective_from"]), effective_to=_typed_date(values["effective_to"]),
        subject_category=Extracted[str](value=category, status=ValueStatus.ACCEPTED,
                                        reason="classified_by_llm" if acc.header else "default"),
        parties=acc.parties, terms=acc.terms, rates=acc.rates, ops=acc.ops, issues=issues)

