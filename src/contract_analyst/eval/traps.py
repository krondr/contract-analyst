"""Interpret golden traps (scenario expectations) against a finished run."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

from pydantic import BaseModel

from contract_analyst.eval.golden import GoldenSet, GoldenTrap
from contract_analyst.eval.metrics import RunsAt, same_value, similar_text
from contract_analyst.kb.store import KnowledgeBase


class TrapResult(BaseModel):
    trap_id: str
    passed: bool
    detail: str


def _check(trap: GoldenTrap, kb: KnowledgeBase, runs: RunsAt) -> tuple[bool, str]:
    e = trap.expected
    records = kb.records()
    chain = next((c for c in kb.chains() if c.contract_number == trap.contract_number), None)
    if trap.kind == "issue_present":
        codes = {i.code for i in kb.get_issues(trap.contract_number)}
        return e["code"] in codes, f"issues: {sorted(codes)}"
    if trap.kind == "status_is":
        result = runs.terms(trap.contract_number, date.fromisoformat(str(e["as_of"])))
        got = result.status if result else None
        return got == e["status"], f"status: {got}"
    if trap.kind == "order_is":
        got = [d.doc_id for d in chain.docs] if chain else []
        return got == e["order"], f"order: {got}"
    if trap.kind == "excluded_from_consolidation":
        doc = next((d for d in (chain.docs if chain else []) if d.doc_id == e["doc_id"]), None)
        return bool(doc and doc.invalidated_by), f"invalidated_by: {doc.invalidated_by if doc else None}"
    if trap.kind == "term_value_at":
        result = runs.terms(trap.contract_number, date.fromisoformat(str(e["as_of"])))
        state = result.terms.get(e["term"]) if result else None
        got = state.current.value if state else None
        return similar_text(e["value"], got), f"value: {got}"
    if trap.kind == "rate_planned":
        result = runs.terms(trap.contract_number, date.fromisoformat(str(e["as_of"])))
        planned = result.planned_rates if result else []
        hit = any(Decimal(str(r.amount)) == Decimal(str(e["amount"]).replace(",", ".")) for r in planned)
        return hit, f"planned: {[str(r.amount) for r in planned]}"
    if trap.kind == "doc_kind_is":
        record = records.get(trap.doc_id or e.get("doc_id"))
        got = record.kind.value if record else None
        return got == e["kind"], f"kind: {got}"
    if trap.kind == "normalized_number":
        record = records.get(trap.doc_id or e.get("doc_id"))
        got = record.number.value if record else None
        return same_value(got, e["number"]), f"number: {got}"
    return False, f"unknown trap kind {trap.kind}"


def check_traps(run_dir: Path, golden: GoldenSet, runs_at: dict[date, Path] | None = None) -> list[TrapResult]:
    kb, runs = KnowledgeBase(Path(run_dir) / "kb"), RunsAt(run_dir, runs_at)
    results = []
    for trap in golden.traps:
        try:
            passed, detail = _check(trap, kb, runs)
        except (KeyError, ValueError, TypeError) as exc:
            passed, detail = False, f"bad trap definition: {exc}"
        results.append(TrapResult(trap_id=trap.trap_id, passed=passed, detail=detail))
    return results
