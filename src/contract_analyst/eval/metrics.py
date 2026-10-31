"""Score a run against the golden set: document kinds, chains, critical values, consolidation, clusters."""

from __future__ import annotations

import json
import re
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path

from pydantic import BaseModel

from contract_analyst.eval.golden import GoldenRate, GoldenSet
from contract_analyst.ingest.normalize import normalize_text
from contract_analyst.kb.store import KnowledgeBase
from contract_analyst.models import DocumentRecord, Extracted, ValueStatus
from contract_analyst.parsing.patterns import DOC_NUMBER_RE

_WORD = re.compile(r"[0-9a-zа-яё]{3,}")
SCALAR_FIELDS = {"number": "number", "doc_date": "doc_date", "effective_from": "effective_from"}


class Scores(BaseModel):
    doc_kind_acc: float = 0.0
    chain_acc: float = 0.0
    critical_total: int = 0
    critical_accepted_correct: int = 0
    critical_accepted_wrong: int = 0
    critical_coverage: float = 0.0
    consolidation_acc: float = 0.0
    clusters_match: bool = False
    traps_passed: int = 0
    traps_total: int = 0
    wall_seconds: float = 0.0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    json_valid_rate: float = 1.0
    tool_call_valid_rate: float = 1.0
    safety_net: list[str] = []


def _norm_text(value) -> str:
    return " ".join(normalize_text(str(value)).casefold().replace("ё", "е").split())


def same_value(a, b) -> bool:
    if a is None or b is None:
        return a is b
    try:
        return Decimal(str(a).replace(",", ".")) == Decimal(str(b).replace(",", "."))
    except InvalidOperation:
        return _norm_text(a) == _norm_text(b)


def similar_text(a: str | None, b: str | None) -> bool:
    wa, wb = set(_WORD.findall(_norm_text(a or ""))), set(_WORD.findall(_norm_text(b or "")))
    if not wa or not wb:
        return _norm_text(a or "") == _norm_text(b or "")
    return len(wa & wb) / min(len(wa), len(wb)) >= 0.6


def _accepted(ext: Extracted):
    if ext.status != ValueStatus.ACCEPTED or ext.value is None:
        return None
    return ext.value.isoformat() if isinstance(ext.value, date) else ext.value


def _rate_set(rates) -> set[tuple]:
    return {(Decimal(r.amount).normalize(), r.valid_from, r.valid_to) for r in rates}


def _op_key(op, target) -> tuple:
    m = DOC_NUMBER_RE.search(target or "")
    return (str(op), m.group(0) if m else None)


class _Tally:
    def __init__(self) -> None:
        self.total = self.correct = self.wrong = 0

    def scalar(self, golden_value, golden_status: str, predicted) -> None:
        if golden_status == "accepted" and golden_value is not None:
            self.total += 1
            if predicted is not None:
                if same_value(golden_value, predicted):
                    self.correct += 1
                else:
                    self.wrong += 1
        elif predicted is not None:
            self.wrong += 1  # value accepted where the golden set has none: a hallucination


def _critical(golden: GoldenSet, records: dict[str, DocumentRecord]) -> _Tally:
    tally = _Tally()
    for g in golden.documents:
        r = records.get(g.doc_id)
        for field in SCALAR_FIELDS:
            gv = getattr(g, field)
            tally.scalar(gv.value, gv.status, _accepted(getattr(r, field)) if r else None)
        parent = r.parent.value if r and r.parent.status == ValueStatus.ACCEPTED else None
        tally.scalar(g.parent_number.value, g.parent_number.status, parent.number if parent else None)
        tally.scalar(g.parent_date.value, g.parent_date.status,
                     parent.date_stated.isoformat() if parent and parent.date_stated else None)
        gold_rates = _rate_set(g.rates)
        got_rates = _rate_set([x for x in (r.rates if r else []) if x.status == ValueStatus.ACCEPTED])
        tally.total += len(gold_rates)
        tally.correct += len(gold_rates & got_rates)
        tally.wrong += len(got_rates - gold_rates)
        gold_ops = {_op_key(o.op, o.target) for o in g.ops}
        got_ops = {_op_key(o.op, o.target) for o in (r.ops if r else []) if o.status == ValueStatus.ACCEPTED}
        tally.total += len(gold_ops)
        tally.correct += len(gold_ops & got_ops)
        tally.wrong += len(got_ops - gold_ops)
    return tally


def _consolidation(golden: GoldenSet, kb: KnowledgeBase) -> float:
    points = earned = 0
    for chain in golden.chains:
        for when in golden.as_of_dates:
            result = kb.get_contract_terms(chain.contract_number, when)
            if when in chain.status_at:
                points += 1
                earned += bool(result and result.status == chain.status_at[when])
            if when in chain.rates_at:
                points += 1
                earned += bool(result and _rate_set(result.rates) == _rate_set(chain.rates_at[when]))
            for term, value in chain.terms_at.get(when, {}).items():
                points += 1
                state = result.terms.get(term) if result else None
                earned += bool(state and similar_text(value, state.current.value))
    return earned / points if points else 1.0


def _usage(run_dir: Path) -> dict:
    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    trace_path = run_dir / "trace.jsonl"
    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()] if trace_path.exists() else []
    calls = sum(1 for e in events if e.get("event") == "tool_call")
    bad = sum(1 for e in events if e.get("event") == "error" and e.get("message") == "bad arguments")
    metrics["tool_call_valid_rate"] = 1 - bad / (calls + bad) if calls + bad else 1.0
    return metrics


def score_run(run_dir: Path, golden: GoldenSet) -> Scores:
    from contract_analyst.eval.traps import check_traps

    kb = KnowledgeBase(Path(run_dir) / "kb")
    records, chains = kb.records(), {c.contract_number: c for c in kb.chains()}
    kinds = [records.get(g.doc_id) is not None and records[g.doc_id].kind.value == g.kind for g in golden.documents]
    chain_ok = []
    for g in golden.chains:
        c = chains.get(g.contract_number)
        chain_ok.append(bool(c and c.main_doc_id == g.main_doc_id and [d.doc_id for d in c.docs] == g.order))
    tally = _critical(golden, records)
    clusters = {(c.subject_category, frozenset(c.members)) for c in kb.clusters()}
    traps = check_traps(Path(run_dir), golden)
    usage = _usage(Path(run_dir))
    llm = usage.get("llm_usage", {})
    invalid = sum(1 for r in records.values() for i in r.issues if i.code == "llm_invalid_json")
    return Scores(
        doc_kind_acc=sum(kinds) / len(kinds) if kinds else 1.0,
        chain_acc=sum(chain_ok) / len(chain_ok) if chain_ok else 1.0,
        critical_total=tally.total, critical_accepted_correct=tally.correct, critical_accepted_wrong=tally.wrong,
        critical_coverage=tally.correct / tally.total if tally.total else 1.0,
        consolidation_acc=_consolidation(golden, kb),
        clusters_match=clusters == {(c.category, frozenset(c.members)) for c in golden.clusters},
        traps_passed=sum(t.passed for t in traps), traps_total=len(traps),
        wall_seconds=usage.get("seconds", 0.0), llm_calls=llm.get("calls", 0),
        input_tokens=llm.get("input_tokens", 0), output_tokens=llm.get("output_tokens", 0),
        json_valid_rate=1 - invalid / llm["calls"] if llm.get("calls") else 1.0,
        tool_call_valid_rate=usage["tool_call_valid_rate"], safety_net=usage.get("safety_net", []))


__all__ = ["GoldenRate", "Scores", "same_value", "score_run", "similar_text"]
