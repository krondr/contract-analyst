"""As-of consolidation: fold a chain's accepted changes, in chain order, into the terms in force on a date."""

from __future__ import annotations

import re
from datetime import date

from contract_analyst.labels import TERM_RU
from contract_analyst.models import (
    LLM_FAILURES,
    AmendmentOp,
    ChainDoc,
    ConsolidatedContract,
    ContractChain,
    DocumentRecord,
    Extracted,
    Issue,
    OpType,
    RateRow,
    Source,
    TermChange,
    TermState,
    TimelineEvent,
    ValueStatus,
)
from contract_analyst.parsing.patterns import _OCR_NUMBER_RE, find_doc_numbers

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

_TARGET_APPENDIX = re.compile(r"Приложени\w*\s*(?:№\s*)?(\d+)", re.IGNORECASE)
_TARGET_RANGE = re.compile(r"(?<![\d.])(\d{1,3})\s*[-–—]\s*(\d{1,3})(?![\d.])")
_TARGET_CLAUSE = re.compile(r"(?<![\d.])\d{1,3}(?:\.\d{1,3})*(?!\d)")


def _parse_target(target: str) -> tuple[str | None, str | None, tuple[int, int] | None] | None:
    """(appendix, clause, range of top-level clauses) named by an operation's target, or None if it names none."""
    text = _OCR_NUMBER_RE.sub(" ", target)
    appendix = None
    if m := _TARGET_APPENDIX.search(text):
        appendix, text = f"Приложение {m.group(1)}", text[:m.start()] + " " + text[m.end():]
    if m := _TARGET_RANGE.search(text):
        return appendix, None, (int(m.group(1)), int(m.group(2)))
    if m := _TARGET_CLAUSE.search(text):
        return appendix, m.group(0), None
    return (appendix, None, None) if appendix else None


def covers(target: str | None, clause: str | None) -> bool | None:
    """Whether an operation on `target` rewrites the clause a value was taken from ("12.2", "Приложение 1/1.6").

    None when either side is unknown: then the caller cannot tell a replacement from an addition.
    """
    if not target or not clause or clause == "preamble" or (parsed := _parse_target(target)) is None:
        return None
    t_appendix, t_clause, t_range = parsed
    c_appendix, _, c_clause = clause.rpartition("/") if "/" in clause else (None, "", clause)
    if c_appendix is None and clause.startswith("Приложение"):
        c_appendix, c_clause = clause, None
    if t_appendix != c_appendix:
        return False
    if t_clause is None and t_range is None:
        return True
    if c_clause is None:
        return False
    if t_range:
        top = c_clause.split(".")[0]
        return top.isdigit() and t_range[0] <= int(top) <= t_range[1]
    return c_clause == t_clause or c_clause.startswith(t_clause + ".")


def _other_document(target: str | None, contract_number: str) -> bool:
    numbers = find_doc_numbers(target or "")
    return bool(numbers) and contract_number not in numbers


def _ops_reliable(rec: DocumentRecord) -> bool:
    """The supplement's operations were read: the ops group answered and the gate kept every operation found."""
    failed = any(i.code in LLM_FAILURES and "группа ops" in i.message for i in rec.issues)
    return not failed and not rec.refused_ops and all(op.status == ValueStatus.ACCEPTED for op in rec.ops)


def _term_mode(clause: str | None, main_ops: list[AmendmentOp], reliable: bool,
               suspensions: list[AmendmentOp] = ()) -> str:
    """How a supplement's value meets the current one: "replace", "combine" or "conflict".

    A value is replaced only when an operation rewrites or suspends the clause it came from; operations on other
    clauses, or none at all, add to it. If the operations could not be read, the change is a conflict, never a silent
    addition.
    """
    verdicts = [covers(op.target, clause) for op in main_ops]
    if True in verdicts or True in [covers(op.target, clause) for op in suspensions]:
        return "replace"
    if not reliable:
        return "conflict"
    if None in verdicts:
        return "replace"
    return "combine"


def _end(rec: DocumentRecord) -> date | None:
    """When a document stops applying: its own end date. An operation's end date limits that operation only."""
    return _accepted(rec.effective_to)


def document_start(rec: DocumentRecord) -> date | None:
    """When a document's changes start: its entry into force, or its signing date if it states no other start.
    A stated start the gate refused leaves it unknown: the signing date would be a guess (spec §6.5)."""
    if rec.effective_from.status == ValueStatus.ABSENT:
        return _accepted(rec.doc_date)
    return _accepted(rec.effective_from)


def _effective(rec: DocumentRecord, chain: ContractChain, doc: ChainDoc) -> date | None:
    if doc.role == "main" and chain.main_date is not None and chain.main_date.value is not None:
        return chain.main_date.value
    return document_start(rec)


def _start_refused(rec: DocumentRecord, doc: ChainDoc) -> bool:
    """A supplement states when it enters into force, but the gate could not confirm that date."""
    return doc.role != "main" and rec.effective_from.status not in (ValueStatus.ACCEPTED, ValueStatus.ABSENT)


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


def _repeat_key(text: str) -> str:
    return " ".join(text.casefold().split()).rstrip(".,;:!? ")


def _repeats(value: str, current: str) -> bool:
    """The value is the current one or one of the values combined into it, up to case, spaces and final punctuation."""
    key = _repeat_key(value)
    return any(key == _repeat_key(part) for part in (current, *current.split("; ")))


class _State:
    def __init__(self) -> None:
        self.terms: dict[str, TermState] = {}
        self.since: dict[str, date | None] = {}
        self.rates: list[RateRow] = []
        self.planned_rates: list[RateRow] = []
        self.issues: list[Issue] = []

    def _retire(self, name: str, state: TermState) -> None:
        if state.current.status == ValueStatus.ACCEPTED:
            state.history.append(TermChange(value=state.current.value, effective_from=self.since.get(name),
                                            doc_id=state.current.source.doc_id if state.current.source else "",
                                            source=state.current.source))

    def reset(self) -> None:
        for name, state in self.terms.items():
            self._retire(name, state)
            state.current = Extracted[str](status=ValueStatus.ABSENT, reason="заменено новой редакцией документа")
        self.rates = []

    def clause_of(self, name: str) -> str | None:
        state = self.terms.get(name)
        return state.current.source.clause_id if state and state.current.source else None

    def set_term(self, name: str, ext: Extracted[str], eff: date | None, doc_id: str, as_of: date,
                 mode: str = "replace") -> None:
        change = TermChange(value=ext.value, effective_from=eff, doc_id=doc_id, source=ext.source)
        state = self.terms.get(name)
        if mode == "undated":  # the document's start was refused: whether the value applies on as_of is unknown
            state = state or self.terms.setdefault(name, TermState(name=name, current=Extracted[str](
                status=ValueStatus.ABSENT)))
            current = state.current
            self._retire(name, state)
            self.since[name] = None
            sources = [s for s in (current.source if current.status == ValueStatus.ACCEPTED else None, ext.source) if s]
            state.current = Extracted[str](status=ValueStatus.CONFLICT, alternatives=sources, reason=(
                f"effective_date_unconfirmed: дата вступления соглашения в силу не подтверждена, поэтому неясно, "
                f"действует ли «{ext.value}» на {as_of}"))
            return
        if eff is not None and eff > as_of:
            if state is None:
                state = self.terms[name] = TermState(name=name, current=Extracted[str](status=ValueStatus.ABSENT))
            state.planned.append(change)
            return
        if state is None:
            self.terms[name], self.since[name] = TermState(name=name, current=ext), eff
            return
        current = state.current
        if mode != "replace" and current.status == ValueStatus.ACCEPTED and _repeats(ext.value or "", current.value or ""):
            return
        if mode == "combine" and current.status == ValueStatus.ACCEPTED:
            state.current = ext.model_copy(update={
                "value": f"{current.value}; {ext.value}",
                "alternatives": [*current.alternatives, *([current.source] if current.source else [])]})
            self.since[name] = eff
            return
        self._retire(name, state)
        self.since[name] = eff
        if mode == "conflict" and current.status == ValueStatus.ACCEPTED:
            sources = [s for s in (current.source, ext.source) if s]
            reason = (f"ops_not_identified: операции соглашения не прочитаны, поэтому неясно, заменяет «{ext.value}» "
                      f"прежнее «{current.value}» или дополняет его")
            state.current = Extracted[str](status=ValueStatus.CONFLICT, reason=reason, alternatives=sources)
            self.issues.append(Issue(code="ops_not_identified", message=f"{TERM_RU.get(name, name)}: {reason}",
                                     sources=sources))
            return
        state.current = ext

    def retire_rates(self, target: str | None, start: date | None, as_of: date) -> None:
        """A clause or appendix in new wording takes its old rates with it: the current ones once the new wording
        starts, the planned ones that would start on or after it. An unknown start retires them now."""
        def old(r: RateRow) -> bool:
            return covers(target, r.source.clause_id if r.source else None) is True

        if start is None or start <= as_of:
            self.rates = [r for r in self.rates if not old(r)]
        self.planned_rates = [r for r in self.planned_rates if not old(r)
                              or (start is not None and r.valid_from is not None and r.valid_from < start)]

    def drop_rate(self, label: str) -> None:
        self.rates = [r for r in self.rates if r.label != label]

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
    known = [d for doc in chain.docs if (rec := records.get(doc.doc_id)) and (d := _effective(rec, chain, doc))]
    start = main_date or (min(known) if known else None)
    if start is None:
        status = "unknown"
        issues.append(Issue(code="no_dates", message="Ни у одного документа цепочки нет принятой даты: "
                                                     "действует ли договор на дату анализа, определить нельзя"))
    elif start > as_of:
        status, since = "not_yet_in_force", start

    for doc in chain.docs:
        rec = records.get(doc.doc_id)
        if rec is None:
            issues.append(Issue(code="missing_record", message=f"Документ {doc.doc_id} не разобран"))
            continue
        eff = _effective(rec, chain, doc)
        refused = _start_refused(rec, doc)
        end = _end(rec) if doc.role != "main" else None
        expired = end is not None and end < as_of
        summary = (_summary(rec, doc) + (f" (истёк {end})" if expired else "")
                   + (" (дата вступления в силу не подтверждена)" if refused else ""))
        timeline.append(TimelineEvent(when=eff, doc_id=doc.doc_id, number=doc.number, summary=summary,
                                      invalidated=doc.invalidated_by is not None))
        if doc.invalidated_by is not None or expired:
            continue
        accepted_ops = [op for op in rec.ops if op.status == ValueStatus.ACCEPTED]
        main_ops = [op for op in accepted_ops
                    if op.op in REPLACING and not _other_document(op.target, chain.contract_number)]
        suspensions = [op for op in accepted_ops
                       if op.op == OpType.SUSPEND_PART and not _other_document(op.target, chain.contract_number)]
        if refused:
            src = rec.effective_from.source
            issues.append(Issue(
                code="effective_date_unconfirmed", sources=[Source(doc_id=doc.doc_id, page=src.page, quote="")] if src
                else [], message=f"{doc.number or doc.doc_id}: дата вступления в силу не подтверждена "
                                 f"({(rec.effective_from.reason or '').split(':')[0]}) — изменённые им условия и ставки "
                                 "не показаны как действующие"))
        restated = any(op.op == OpType.RESTATE_DOCUMENT for op in main_ops)
        if restated and not refused and (eff is None or eff <= as_of):
            state.reset()
        reliable = _ops_reliable(rec)
        for name, ext in rec.terms.items():
            if ext.status == ValueStatus.ACCEPTED:
                mode = ("undated" if refused else "replace" if doc.role == "main" or restated
                        else _term_mode(state.clause_of(name), main_ops, reliable, suspensions))
                state.set_term(name, ext, eff, doc.doc_id, as_of, mode=mode)
        for op in main_ops:
            state.retire_rates(op.target, None if refused else eff, as_of)
        rows = [row for row in rec.rates if row.status == ValueStatus.ACCEPTED]
        for row in sorted(rows, key=lambda r: r.valid_from or eff or date.min):
            if refused and row.valid_from is None:  # its start is the refused date: neither it nor the old rate holds
                state.drop_rate(row.label)
            else:
                state.add_rate(row, eff, as_of)
        for op in accepted_ops:  # ending another document does not end the contract
            when = op.effective_from or eff
            if op.op not in (OpType.TERMINATE, OpType.SUSPEND_PART) or _other_document(op.target, chain.contract_number):
                continue
            if when is None and refused:  # when it takes effect is unknown, so is whether it already has
                if status != "terminated":
                    status, since = "unknown", None
            elif when is not None and when > as_of:
                timeline.append(TimelineEvent(when=when, doc_id=doc.doc_id, number=doc.number,
                                              summary=f"запланировано: {_OP_LABELS[op.op]}"))
            elif op.op == OpType.TERMINATE:
                status, since = "terminated", when
            elif status != "terminated":
                status, since = "partially_suspended", when

    for name, term in state.terms.items():
        term.since = state.since.get(name) if term.current.status == ValueStatus.ACCEPTED else None
    return ConsolidatedContract(contract_number=chain.contract_number, as_of=as_of, status=status, status_since=since,
                                terms=state.terms, rates=state.rates, planned_rates=state.planned_rates,
                                timeline=timeline, issues=issues + state.issues)
