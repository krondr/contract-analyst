"""E3 alternative: the LLM folds the chain itself. Code builds the timeline; the LLM decides current values."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel

from contract_analyst.assemble.consolidate import consolidate
from contract_analyst.llm.client import LLMClient
from contract_analyst.llm.prompts import load_prompt
from contract_analyst.llm.structured import StructuredFailure, call_structured
from contract_analyst.models import (
    ConsolidatedContract,
    ContractChain,
    DocumentRecord,
    Extracted,
    Issue,
    RateRow,
    Source,
    TermState,
    ValueStatus,
)
from contract_analyst.parsing.patterns import parse_money


class MergedTerm(BaseModel):
    name: str
    value: str
    doc_id: str


class MergedRate(BaseModel):
    label: str
    amount: str
    valid_from: str | None = None
    valid_to: str | None = None
    doc_id: str


class MergedOut(BaseModel):
    status: Literal["active", "not_yet_in_force", "partially_suspended", "terminated", "unknown"]
    status_since: str | None = None
    terms: list[MergedTerm] = []
    rates: list[MergedRate] = []
    planned_rates: list[MergedRate] = []


def _facts(chain: ContractChain, records: dict[str, DocumentRecord], as_of: date) -> str:
    lines = [f"Договор {chain.contract_number}. Дата свёртки: {as_of}."]
    for d in chain.docs:
        r = records.get(d.doc_id)
        if r is None:
            continue
        mark = " (ОТМЕНЁН)" if d.invalidated_by else ""
        eff = r.effective_from.value or r.doc_date.value
        lines.append(f"\nДокумент doc_id={d.doc_id} № {d.number or '—'} ({d.role}){mark}, действует с {eff or '—'}:")
        lines += [f"- {k}: {v.value}" for k, v in r.terms.items() if v.status == ValueStatus.ACCEPTED]
        lines += [f"- ставка «{x.label}»: {x.amount} с {x.valid_from or '—'} по {x.valid_to or '—'}"
                  for x in r.rates if x.status == ValueStatus.ACCEPTED]
        lines += [f"- операция {o.op.value} {o.target or ''} с {o.effective_from or '—'}"
                  for o in r.ops if o.status == ValueStatus.ACCEPTED]
    return "\n".join(lines)


def _iso(value: str | None) -> date | None:
    try:
        return date.fromisoformat(value) if value else None
    except ValueError:
        return None


def _rate(m: MergedRate, known: set[str]) -> RateRow:
    amount = parse_money(m.amount)
    ok = amount is not None and m.doc_id in known
    return RateRow(label=m.label, amount=amount or 0, valid_from=_iso(m.valid_from), valid_to=_iso(m.valid_to),
                   status=ValueStatus.ACCEPTED if ok else ValueStatus.REJECTED,
                   source=Source(doc_id=m.doc_id, page=1, quote=m.amount) if ok else None,
                   reason=None if ok else "no_source: модель не указала документ-источник")


def consolidate_llm(chain: ContractChain, records: dict[str, DocumentRecord], as_of: date,
                    client: LLMClient) -> ConsolidatedContract:
    base = consolidate(chain, records, as_of)
    messages = [{"role": "system", "content": load_prompt("merge_terms").render()},
                {"role": "user", "content": _facts(chain, records, as_of)}]
    out = call_structured(client, messages, MergedOut)
    if isinstance(out, StructuredFailure):
        return base.model_copy(update={"issues": [*base.issues, Issue(code="llm_merge_failed",
                                                                       message="модель не свела условия, использована свёртка кодом")]})
    known = {d.doc_id for d in chain.docs}
    terms = {}
    for t in out.terms:
        ok = t.doc_id in known
        current = Extracted[str](value=t.value, status=ValueStatus.ACCEPTED if ok else ValueStatus.REJECTED,
                                 source=Source(doc_id=t.doc_id, page=1, quote=t.value) if ok else None,
                                 reason=None if ok else "no_source: модель не указала документ-источник")
        terms[t.name] = TermState(name=t.name, current=current)
    return base.model_copy(update={"status": out.status, "status_since": _iso(out.status_since), "terms": terms,
                                   "rates": [_rate(r, known) for r in out.rates],
                                   "planned_rates": [_rate(r, known) for r in out.planned_rates]})
