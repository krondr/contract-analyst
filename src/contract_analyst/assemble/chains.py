"""Group documents into contract chains: link by numbers in the text, order by effective dates."""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import date

from contract_analyst.models import (
    ChainDoc,
    ContractChain,
    DocKind,
    DocumentRecord,
    Extracted,
    Issue,
    OpType,
    ValueStatus,
)
from contract_analyst.parsing.patterns import find_doc_numbers

_GOOD = (ValueStatus.ACCEPTED, ValueStatus.CONFLICT)
_YEAR_RE = re.compile(r"^D(\d{2})")
_PARTY_NOISE = re.compile(r"[\"'«»“”„\s]")


def _value(ext: Extracted) -> object | None:
    return ext.value if ext.status in _GOOD else None


def _parent_number(rec: DocumentRecord) -> str | None:
    parent = _value(rec.parent)
    return parent.number if parent else None


def _order(rec: DocumentRecord) -> tuple[str, str]:
    """Sort key: effective date, then signing date, then number; and what the order rests on."""
    number = _value(rec.number) or rec.doc_id
    signed = _value(rec.doc_date)
    signed_key = signed.isoformat() if signed is not None else "9999-99-99"
    if (eff := _value(rec.effective_from)) is not None:
        return f"{eff.isoformat()}|{signed_key}|{number}", "effective_from"
    if signed is not None:
        return f"{signed_key}|{signed_key}|{number}", "doc_date"
    m = _YEAR_RE.match(str(number))
    year = 2000 + int(m.group(1)) if m else 9999
    return f"{year}-99-99|9999-99-99|{number}", "number"


def _is_root(rec: DocumentRecord) -> bool:
    number, parent = _value(rec.number), _parent_number(rec)
    if number and parent == number:
        return True  # document restates its own contract: it is the base text
    return parent is None and rec.kind.value in (DocKind.MAIN, DocKind.FRAMEWORK, DocKind.UNKNOWN) and bool(number)


def _party_key(name: str) -> str:
    return _PARTY_NOISE.sub("", name).casefold()


def _names(rec: DocumentRecord) -> set[str]:
    return {_party_key(p.name.value) for p in rec.parties if p.name.status == ValueStatus.ACCEPTED and p.name.value}


def _main_date(main: DocumentRecord | None, sups: list[DocumentRecord], issues: list[Issue]) -> Extracted[date] | None:
    stated = [(r.parent.value.date_stated, r.parent.source) for r in sups
              if r.parent.status in _GOOD and r.parent.value and r.parent.value.date_stated]
    own = main.doc_date if main and main.doc_date.status == ValueStatus.ACCEPTED else None
    if own is not None:
        others = [src for d, src in stated if d != own.value and src]
        if others:
            issues.append(Issue(code="parent_date_conflict",
                                message=f"Дата договора {own.value} не совпадает с датой в ссылках дополнительных соглашений",
                                sources=[s for s in [own.source, *others] if s]))
            return own.model_copy(update={"status": ValueStatus.CONFLICT, "alternatives": others,
                                          "reason": "дата в договоре и в ссылках ДС различается"})
        return own
    if not stated:
        return None
    distinct = {d for d, _ in stated}
    first_src = next((s for _, s in stated if s), None)
    if len(distinct) == 1:
        (value,) = distinct
        issues.append(Issue(code="date_from_references",
                            message=f"Дата договора {value} взята из ссылок дополнительных соглашений",
                            sources=[s for _, s in stated if s]))
        return Extracted[date](value=value, status=ValueStatus.ACCEPTED, source=first_src,
                               reason="inferred_from_references")
    issues.append(Issue(code="parent_date_conflict", message="Дополнительные соглашения ссылаются на разные даты договора",
                        sources=[s for _, s in stated if s]))
    return Extracted[date](value=min(distinct), status=ValueStatus.CONFLICT, source=first_src,
                           alternatives=[s for _, s in stated if s], reason="разные даты в ссылках ДС")


def _root_number(number: str, parent_of: dict[str, str]) -> str:
    """Follow parent links from a supplement's number up to the contract its chain belongs to."""
    seen: set[str] = set()
    while number in parent_of and number not in seen:
        seen.add(number)
        number = parent_of[number]
    return number


def _main_date_checked(main: DocumentRecord | None, sups: list[DocumentRecord], issues: list[Issue]) -> Extracted[date] | None:
    """The contract's date, plus a flag for every supplement dated before it: a supplement cannot precede its contract,
    so either a date or the link is wrong."""
    main_date = _main_date(main, sups, issues)
    if main_date is None or main_date.value is None:
        return main_date
    for r in sups:
        when = _value(r.effective_from) or _value(r.doc_date)
        if when is not None and when < main_date.value:
            issues.append(Issue(code="supplement_before_main",
                                message=f"{_value(r.number) or r.file_name} датировано {when}, раньше договора "
                                        f"({main_date.value}): неверна дата или связь с договором",
                                sources=[s for s in [r.effective_from.source or r.doc_date.source] if s]))
    return main_date


def build_chains(records: list[DocumentRecord]) -> list[ContractChain]:
    by_number: dict[str, DocumentRecord] = {}
    for r in records:
        if (number := _value(r.number)) is not None:
            by_number.setdefault(number, r)
    parent_of = {number: parent for r in records if not _is_root(r)
                 and (number := _value(r.number)) and (parent := _parent_number(r)) and parent != number}
    roots: dict[str, list[DocumentRecord]] = defaultdict(list)
    members: dict[str, list[DocumentRecord]] = defaultdict(list)
    amends: dict[str, list[tuple[DocumentRecord, str]]] = defaultdict(list)
    loose: list[DocumentRecord] = []
    for r in records:
        if _is_root(r):
            roots[_value(r.number)].append(r)
        elif (parent := _parent_number(r)) is not None:
            root = _root_number(parent, parent_of)
            members[root].append(r)
            if root != parent:
                amends[root].append((r, parent))
        else:
            loose.append(r)

    chains: list[ContractChain] = []
    for key in sorted(set(roots) | set(members)):
        issues: list[Issue] = [
            Issue(code="amends_supplement", message=f"{_value(r.number) or r.file_name} изменяет соглашение {parent}, "
                                                    f"а не сам договор {key}", sources=[s for s in [r.parent.source] if s])
            for r, parent in amends.get(key, [])]
        candidates = sorted(roots.get(key, []), key=lambda r: (r.is_draft, _value(r.doc_date) or date.max, r.doc_id))
        main = candidates[0] if candidates else None
        if len(candidates) > 1:
            issues.append(Issue(code="duplicate_main", message=f"Несколько документов претендуют на основной договор {key}"))
        sups = sorted(members.get(key, []) + candidates[1:], key=lambda r: _order(r)[0])
        if main is None:
            issues.append(Issue(code="missing_main", message=f"Основного договора {key} нет в пакете"))
        elif main.is_draft:
            issues.append(Issue(code="draft_main", message=f"Основной договор {key} — проект, подписание не подтверждено"))

        docs: list[ChainDoc] = []
        if main is not None:
            mk, mb = _order(main)
            docs.append(ChainDoc(doc_id=main.doc_id, number=_value(main.number), role="main", order_key=mk, order_basis=mb))
        for r in sups:
            k, b = _order(r)
            docs.append(ChainDoc(doc_id=r.doc_id, number=_value(r.number), role="supplement", order_key=k, order_basis=b))
        if any(d.order_basis == "number" and d.role == "supplement" for d in docs):
            issues.append(Issue(code="order_by_number", message="Порядок части соглашений определён по номеру: дат нет"))

        chain_records = [r for r in [main, *sups] if r is not None]
        by_doc = {d.number: d for d in docs if d.number}
        for r in chain_records:
            for op in r.ops:
                if op.status != ValueStatus.ACCEPTED or not op.target:
                    continue
                targets = find_doc_numbers(op.target)
                target = targets[0] if targets else None
                # Terminating a supplement, not the contract itself, cancels that supplement like an invalidation.
                cancels = op.op == OpType.INVALIDATE_DOCUMENT or (op.op == OpType.TERMINATE and key not in targets)
                if cancels and target and target in by_doc:
                    by_doc[target].invalidated_by = r.doc_id
                    done = "признан недействительным" if op.op == OpType.INVALIDATE_DOCUMENT else "расторгнут"
                    issues.append(Issue(code="invalidated", message=f"{target} {done} документом {_value(r.number)}",
                                        sources=[s for s in [op.source] if s]))
                elif target and target not in by_number:
                    issues.append(Issue(code="target_missing",
                                        message=f"{_value(r.number)} изменяет {target}, которого нет в пакете",
                                        sources=[s for s in [op.source] if s]))

        main_names = _names(main) if main else set()
        for r in sups:
            extra = _names(r) - main_names
            if main_names and extra:
                issues.append(Issue(code="party_name_mismatch",
                                    message=f"Стороны в {_value(r.number) or r.file_name} не совпадают с основным договором",
                                    sources=[p.name.source for p in r.parties if p.name.source]))
        for r in chain_records:
            issues.extend(r.issues)
        chains.append(ContractChain(contract_number=key, main_doc_id=main.doc_id if main else None,
                                    main_date=_main_date_checked(main, sups, issues), docs=docs, issues=issues))

    for r in loose:
        chains.append(ContractChain(contract_number=_value(r.number) or r.doc_id, main_doc_id=None, docs=[
            ChainDoc(doc_id=r.doc_id, number=_value(r.number), role="unknown", order_key=_order(r)[0],
                     order_basis=_order(r)[1])],
            issues=[Issue(code="unlinked", message=f"{r.file_name}: не удалось связать с договором"), *r.issues]))
    return chains

