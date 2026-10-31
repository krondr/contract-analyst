"""As-of consolidation: fold a chain's accepted changes, in chain order, into the terms in force on a date."""

from __future__ import annotations

from datetime import date

from contract_analyst.labels import TERM_RU
from contract_analyst.models import (
    ChainDoc,
    ConsolidatedContract,
    ContractChain,
    DocumentRecord,
    Extracted,
    Issue,
    OpType,
    RateRow,
    TermChange,
    TermState,
    TimelineEvent,
    ValueStatus,
)

_OP_LABELS = {
    OpType.REPLACE_CLAUSE: "новая редакция пункта", OpType.ADD_CLAUSE: "новый пункт",
    OpType.DELETE_CLAUSE: "исключён пункт", OpType.RESTATE_DOCUMENT: "договор изложен в новой редакции",
    OpType.RESTATE_SECTION: "раздел изложен в новой редакции", OpType.REPLACE_APPENDIX: "новая редакция приложения",
    OpType.TERMINATE: "расторжение", OpType.SUSPEND_PART: "прекращение части обязательств",
    OpType.INVALIDATE_DOCUMENT: "отмена ранее подписанного соглашения", OpType.OTHER: "изменение",
}


def _accepted(ext: Extracted) -> object | None:
    return ext.value if ext.status == ValueStatus.ACCEPTED else None


REPLACING = {OpType.REPLACE_CLAUSE, OpType.RESTATE_SECTION, OpType.REPLACE_APPENDIX, OpType.RESTATE_DOCUMENT,
             OpType.DELETE_CLAUSE}


def _end(rec: DocumentRecord) -> date | None:
    """When a document stops applying: its own end date, else the latest end date of its operations."""
    if (own := _accepted(rec.effective_to)) is not None:
        return own
    ends = [op.effective_to for op in rec.ops if op.status == ValueStatus.ACCEPTED]
    return max(ends) if ends and all(ends) else None


def _effective(rec: DocumentRecord, chain: ContractChain, doc: ChainDoc) -> date | None:
    if doc.role == "main" and chain.main_date is not None and chain.main_date.value is not None:
        return chain.main_date.value
    return _accepted(rec.effective_from) or _accepted(rec.doc_date)


def _summary(rec: DocumentRecord, doc: ChainDoc) -> str:
    if doc.role == "main":
        return "основной договор"
    parts = [f"{_OP_LABELS[op.op]}{f' {op.target}' if op.target else ''}" for op in rec.ops
             if op.status == ValueStatus.ACCEPTED]
    changed = [TERM_RU.get(k, k).lower() for k, v in rec.terms.items() if v.status == ValueStatus.ACCEPTED]
    if rec.rates:
        changed.append("ставки")
    if changed:
        parts.append("меняет: " + ", ".join(changed))
    return "; ".join(parts) or "дополнительное соглашение"


class _State:
    def __init__(self) -> None:
        self.terms: dict[str, TermState] = {}
        self.rates: list[RateRow] = []
        self.planned_rates: list[RateRow] = []

    def reset(self) -> None:
        for state in self.terms.values():
            if state.current.status == ValueStatus.ACCEPTED:
                state.history.append(TermChange(value=state.current.value, effective_from=None,
                                                doc_id=state.current.source.doc_id if state.current.source else "",
                                                source=state.current.source))
            state.current = Extracted[str](status=ValueStatus.ABSENT, reason="заменено новой редакцией документа")
        self.rates = []

    def set_term(self, name: str, ext: Extracted[str], eff: date | None, doc_id: str, as_of: date,
                 combine: bool = False) -> None:
        change = TermChange(value=ext.value, effective_from=eff, doc_id=doc_id, source=ext.source)
        state = self.terms.get(name)
        if eff is not None and eff > as_of:
            if state is None:
                state = self.terms[name] = TermState(name=name, current=Extracted[str](status=ValueStatus.ABSENT))
            state.planned.append(change)
            return
        if state is None:
            self.terms[name] = TermState(name=name, current=ext)
            return
        if combine and state.current.status == ValueStatus.ACCEPTED:
            if ext.value not in (state.current.value or ""):
                state.current = ext.model_copy(update={
                    "value": f"{state.current.value}; {ext.value}",
                    "alternatives": [*state.current.alternatives, *([state.current.source] if state.current.source else [])]})
            return
        if state.current.status == ValueStatus.ACCEPTED:
            state.history.append(TermChange(value=state.current.value, effective_from=None,
                                            doc_id=state.current.source.doc_id if state.current.source else "",
                                            source=state.current.source))
        state.current = ext

    def add_rate(self, row: RateRow, eff: date | None, as_of: date) -> None:
        start = row.valid_from or eff
        if start is not None and start > as_of:
            self.planned_rates.append(row)
        elif row.valid_to is not None and row.valid_to < as_of:
            return
        else:
            self.rates = [r for r in self.rates if r.label != row.label] + [row]


def consolidate(chain: ContractChain, records: dict[str, DocumentRecord], as_of: date) -> ConsolidatedContract:
    state = _State()
    issues: list[Issue] = []
    timeline: list[TimelineEvent] = []
    status, since = "active", None
    if chain.main_doc_id is None:
        issues.append(Issue(code="no_base", message="Основного договора нет: известны только условия из соглашений"))
    main_date = chain.main_date.value if chain.main_date is not None else None
    if main_date is not None and main_date > as_of:
        status, since = "not_yet_in_force", main_date

    for doc in chain.docs:
        rec = records.get(doc.doc_id)
        if rec is None:
            issues.append(Issue(code="missing_record", message=f"Документ {doc.doc_id} не разобран"))
            continue
        eff = _effective(rec, chain, doc)
        end = _end(rec) if doc.role != "main" else None
        expired = end is not None and end < as_of
        summary = _summary(rec, doc) + (f" (истёк {end})" if expired else "")
        timeline.append(TimelineEvent(when=eff, doc_id=doc.doc_id, number=doc.number, summary=summary,
                                      invalidated=doc.invalidated_by is not None))
        if doc.invalidated_by is not None or expired:
            continue
        accepted_ops = [op for op in rec.ops if op.status == ValueStatus.ACCEPTED]
        additive = doc.role == "supplement" and not any(op.op in REPLACING for op in accepted_ops)
        if any(op.op == OpType.RESTATE_DOCUMENT for op in accepted_ops) and (eff is None or eff <= as_of):
            state.reset()
        for name, ext in rec.terms.items():
            if ext.status == ValueStatus.ACCEPTED:
                state.set_term(name, ext, eff, doc.doc_id, as_of, combine=additive)
        for row in rec.rates:
            if row.status == ValueStatus.ACCEPTED:
                state.add_rate(row, eff, as_of)
        for op in accepted_ops:
            when = op.effective_from or eff
            if op.op not in (OpType.TERMINATE, OpType.SUSPEND_PART):
                continue
            if when is not None and when > as_of:
                timeline.append(TimelineEvent(when=when, doc_id=doc.doc_id, number=doc.number,
                                              summary=f"запланировано: {_OP_LABELS[op.op]}"))
            elif op.op == OpType.TERMINATE:
                status, since = "terminated", when
            elif status != "terminated":
                status, since = "partially_suspended", when

    return ConsolidatedContract(contract_number=chain.contract_number, as_of=as_of, status=status, status_since=since,
                                terms=state.terms, rates=state.rates, planned_rates=state.planned_rates,
                                timeline=timeline, issues=issues)
