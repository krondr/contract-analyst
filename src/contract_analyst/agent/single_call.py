"""Baseline S: one LLM call per folder with all texts, no gate, no code consolidation."""

from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from contract_analyst.agent import steps
from contract_analyst.agent.context import RunContext
from contract_analyst.agent.runner import DEFAULT_LLM_CACHE, DEFAULT_READ_CACHE, RunResult, _run_dir
from contract_analyst.agent.trace import TraceLogger
from contract_analyst.assemble.chains import build_chains
from contract_analyst.config import Settings
from contract_analyst.llm.client import LLMClient
from contract_analyst.llm.factory import get_llm_client
from contract_analyst.llm.prompts import load_prompt
from contract_analyst.llm.structured import StructuredFailure, call_structured
from contract_analyst.models import (
    ConsolidatedContract,
    DocKind,
    DocumentRecord,
    Extracted,
    Issue,
    ParentRef,
    RateRow,
    TermState,
    ValueStatus,
)
from contract_analyst.ocr import OcrEngine, get_ocr_engine
from contract_analyst.parsing.patterns import parse_money

Status = Literal["active", "not_yet_in_force", "partially_suspended", "terminated", "unknown"]


class SingleDoc(BaseModel):
    file: str
    number: str | None = None
    kind: DocKind = DocKind.UNKNOWN
    parent_number: str | None = None
    doc_date: str | None = None
    effective_from: str | None = None


class SingleRate(BaseModel):
    label: str
    amount: str
    valid_from: str | None = None
    valid_to: str | None = None


class SingleContract(BaseModel):
    number: str
    status: Status = "unknown"
    terms: dict[str, str] = {}
    rates: list[SingleRate] = []


class SingleOut(BaseModel):
    documents: list[SingleDoc] = []
    contracts: list[SingleContract] = []


def _acc(value) -> dict:
    return {"value": value, "status": ValueStatus.ACCEPTED} if value else {"status": ValueStatus.ABSENT}


def _iso(value: str | None) -> date | None:
    try:
        return date.fromisoformat(value) if value else None
    except ValueError:
        return None


def _record(doc_id: str, d: SingleDoc) -> DocumentRecord:
    parent = Extracted[ParentRef](value=ParentRef(number=d.parent_number), status=ValueStatus.ACCEPTED) if d.parent_number \
        else Extracted[ParentRef](status=ValueStatus.ABSENT)
    return DocumentRecord(doc_id=doc_id, file_name=d.file, title=d.file, kind=_acc(d.kind), number=_acc(d.number),
                          doc_date=_acc(_iso(d.doc_date)), city=_acc(None), parent=parent,
                          effective_from=_acc(_iso(d.effective_from)), subject_category=_acc("other"))


def _contract(c: SingleContract, as_of: date) -> ConsolidatedContract:
    return ConsolidatedContract(
        contract_number=c.number, as_of=as_of, status=c.status,
        terms={k: TermState(name=k, current=Extracted[str](value=v, status=ValueStatus.ACCEPTED)) for k, v in c.terms.items()},
        rates=[RateRow(label=r.label, amount=parse_money(r.amount) or 0, valid_from=_iso(r.valid_from),
                       valid_to=_iso(r.valid_to), status=ValueStatus.ACCEPTED) for r in c.rates])


def run_single_call(input_dir: Path, settings: Settings, as_of: date, out_dir: Path, client: LLMClient | None = None,
                    ocr_engine: OcrEngine | None = None, read_cache: Path | None = DEFAULT_READ_CACHE,
                    llm_cache: Path = DEFAULT_LLM_CACHE, console: bool = False) -> RunResult:
    run_dir = _run_dir(out_dir, "S")
    client = client or get_llm_client(settings, llm_cache)
    ocr_engine = ocr_engine or get_ocr_engine(settings.ocr_engine)
    trace = TraceLogger(run_dir, run_dir.name, console=console)
    ctx = RunContext(settings=settings, client=client, ocr_engine=ocr_engine, trace=trace, input_dir=input_dir,
                     run_dir=run_dir, as_of=as_of, read_cache=read_cache)
    started = time.monotonic()
    steps.ingest_all(ctx)
    groups: dict[str, list] = {}
    for doc in ctx.documents.values():
        rel = Path(doc.file.path).relative_to(input_dir) if Path(doc.file.path).is_relative_to(input_dir) else Path(doc.file.name)
        groups.setdefault(rel.parts[0] if len(rel.parts) > 1 else ".", []).append(doc)
    budget = max(4000, (settings.llm_num_ctx - 3000) * 3)
    fields = [f for g in ("core_terms", "dealer") for f in settings.field_groups.get(g, [])]
    system = load_prompt("single_call").render(as_of=as_of, fields=[(f, settings.field_descriptions.get(f, f)) for f in fields])
    contracts: dict[str, ConsolidatedContract] = {}
    issues: list[Issue] = []
    for name, docs in sorted(groups.items()):
        text = "\n\n".join(f"=== файл: {d.file.name} ===\n" + "\n".join(p.text for p in d.pages) for d in docs)[:budget]
        out = call_structured(client, [{"role": "system", "content": system}, {"role": "user", "content": text}], SingleOut)
        trace.event("step", "single_call", group=name, ok=not isinstance(out, StructuredFailure))
        if isinstance(out, StructuredFailure):
            issues.append(Issue(code="llm_invalid_json", message=f"папка {name}: ответ не прошёл схему"))
            continue
        by_name = {d.file.name: d.doc_id for d in docs}
        for d in out.documents:
            if d.file in by_name:
                ctx.records[by_name[d.file]] = _record(by_name[d.file], d)
        for c in out.contracts:
            contracts[c.number] = _contract(c, as_of)
    ctx.chains = build_chains(list(ctx.records.values()))
    ctx.consolidated = contracts
    steps.write_report(ctx)
    seconds = round(time.monotonic() - started, 1)
    metrics = {"run_id": run_dir.name, "variant": "S", "as_of": as_of.isoformat(), "seconds": seconds,
               "model": client.model_name, "llm_usage": dict(client.usage), "safety_net": [],
               "issues": [i.code for i in issues]}
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    return RunResult(run_dir=run_dir, report_path=ctx.report_path, kb_dir=run_dir / "kb", trace_path=trace.path,
                     variant="S", seconds=seconds, llm_usage=dict(client.usage))
