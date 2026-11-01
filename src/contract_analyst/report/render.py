"""Markdown report: structure from a template, every value with a source reference."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import jinja2
from pydantic import BaseModel

from contract_analyst.labels import KIND_RU, STATUS_RU, TERM_RU, VAT_RU
from contract_analyst.models import (
    Cluster,
    ConsolidatedContract,
    ContractChain,
    Decision,
    DocumentRecord,
    DocumentText,
    Extracted,
    Source,
    ValueStatus,
)
from contract_analyst.report.verify import grounding


def _neutralize(value: object) -> object:
    """Every value the template prints comes from a document or the model. It must not open an HTML comment,
    which hides the rest of the report, nor fake the <!-- llm --> markers that verify_claims relies on."""
    if isinstance(value, jinja2.Undefined):
        return value
    return str(value).replace("<", "&lt;").replace(">", "&gt;")


_ENV = jinja2.Environment(loader=jinja2.FileSystemLoader(Path(__file__).parent / "templates"),
                          undefined=jinja2.StrictUndefined, trim_blocks=True, lstrip_blocks=True, autoescape=False,
                          finalize=_neutralize)


class ReflectionItem(BaseModel):
    item: str
    ok: bool
    note: str


class ReportContext(BaseModel):
    run_id: str
    as_of: date
    model: str
    prompt_versions: dict[str, int]
    ocr: str
    documents: list[DocumentText]
    records: dict[str, DocumentRecord]
    chains: list[ContractChain]
    consolidated: dict[str, ConsolidatedContract]
    clusters: list[Cluster]
    decisions: list[Decision]
    reflection: list[ReflectionItem]
    unprocessed: list[dict]
    trace_path: str
    metrics_path: str | None


def ref(source: Source | None) -> str:
    return f"[doc:{source.doc_id}:p{source.page}]" if source else ""


def money(amount: Decimal) -> str:
    whole, _, frac = format(amount.normalize(), "f").partition(".")
    grouped = f"{int(whole):,}".replace(",", " ")
    return f"{grouped},{frac}" if frac else grouped


def value(ext: Extracted | None) -> str:
    if ext is None or ext.status != ValueStatus.ACCEPTED or ext.value is None:
        return "—"
    v = ext.value
    return v.isoformat() if isinstance(v, date) else str(v)


def cell(text: object) -> str:
    return str(text).replace("|", "/").replace("\n", " ")


def _refusals(ctx: ReportContext) -> list[dict]:
    rows = []
    for doc_id, r in sorted(ctx.records.items(), key=lambda kv: kv[1].file_name):
        fields = {f: getattr(r, f) for f in ("number", "doc_date", "city", "parent", "effective_from")}
        fields.update({f"terms.{k}": v for k, v in r.terms.items()})
        for name, ext in fields.items():
            if ext.status in (ValueStatus.REJECTED, ValueStatus.CONFLICT, ValueStatus.REDACTED):
                shown = ext.value.number if hasattr(ext.value, "number") else ext.value
                rows.append({"file": r.file_name, "field": name, "value": shown if shown is not None else "—",
                             "status": ext.status.value, "reason": ext.reason or "", "ref": ref(ext.source)})
        for rate in r.rates:
            if rate.status != ValueStatus.ACCEPTED:
                rows.append({"file": r.file_name, "field": f"ставка «{rate.label}»", "value": money(rate.amount),
                             "status": rate.status.value, "reason": rate.reason or "", "ref": ref(rate.source)})
        for issue in r.issues:
            if issue.code in ("llm_invalid_json", "llm_timeout"):
                rows.append({"file": r.file_name, "field": "ответ модели", "value": "—", "status": issue.code,
                             "reason": issue.message, "ref": ""})
        for op in r.ops:
            if op.status != ValueStatus.ACCEPTED:
                rows.append({"file": r.file_name, "field": f"операция {op.op.value}", "value": op.target or "—",
                             "status": op.status.value, "reason": op.reason or "", "ref": ref(op.source)})
    return rows


def _rate_history(chain: ContractChain, records: dict[str, DocumentRecord]) -> dict[str, list[str]]:
    history: dict[str, list[str]] = {}
    for d in chain.docs:
        record = records.get(d.doc_id)
        if d.invalidated_by or record is None:
            continue
        for rate in record.rates:
            if rate.status == ValueStatus.ACCEPTED:
                since = f" с {rate.valid_from}" if rate.valid_from else ""
                history.setdefault(rate.label, []).append(f"{money(rate.amount)}{since} {ref(rate.source)}")
    return {label: steps for label, steps in history.items() if len(steps) > 1}


def report_sources(ctx: ReportContext) -> dict[str, str]:
    """What backs the LLM-written sentences of this report: see verify.grounding."""
    return grounding(((d.doc_id, p.number, p.text) for d in ctx.documents for p in d.pages), ctx.records, ctx.chains,
                     ctx.consolidated)


def render_report(ctx: ReportContext) -> str:
    docs_by_id = {d.doc_id: d for d in ctx.documents}
    rate_history = {c.contract_number: _rate_history(c, ctx.records) for c in ctx.chains}
    template = _ENV.get_template("report.md.j2")
    return template.render(
        ctx=ctx, docs_by_id=docs_by_id, refusals=_refusals(ctx), rate_history=rate_history, ref=ref, money=money, value=value, cell=cell,
        STATUS_RU=STATUS_RU, VAT_RU=VAT_RU, KIND_RU=KIND_RU, TERM_RU=TERM_RU,
        chains=sorted(ctx.chains, key=lambda c: c.contract_number),
        documents=sorted(ctx.documents, key=lambda d: d.file.name)).rstrip() + "\n"
