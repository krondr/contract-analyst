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
from contract_analyst.parsing.patterns import DOC_NUMBER_RE

_GOOD = (ValueStatus.ACCEPTED, ValueStatus.CONFLICT)
_YEAR_RE = re.compile(r"^D(\d{2})")
_PARTY_NOISE = re.compile(r"[\"'«»“”„\s]")


def _value(ext: Extracted) -> object | None:
    return ext.value if ext.status in _GOOD else None


def _parent_number(rec: DocumentRecord) -> str | None:
    parent = _value(rec.parent)
    return parent.number if parent else None


def _order(rec: DocumentRecord) -> tuple[str, str]:
    number = _value(rec.number) or rec.doc_id
    if (eff := _value(rec.effective_from)) is not None:
        return f"{eff.isoformat()}|{number}", "effective_from"
    if (signed := _value(rec.doc_date)) is not None:
        return f"{signed.isoformat()}|{number}", "doc_date"
    m = _YEAR_RE.match(str(number))
    year = 2000 + int(m.group(1)) if m else 9999
    return f"{year}-99-99|{number}", "number"


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


def build_chains(records: list[DocumentRecord]) -> list[ContractChain]:
    by_number: dict[str, DocumentRecord] = {}
    for r in records:
        if (number := _value(r.number)) is not None:
            by_number.setdefault(number, r)
    roots: dict[str, list[DocumentRecord]] = defaultdict(list)
    members: dict[str, list[DocumentRecord]] = defaultdict(list)
    loose: list[DocumentRecord] = []
    for r in records:
        if _is_root(r):
            roots[_value(r.number)].append(r)
        elif (parent := _parent_number(r)) is not None:
            members[parent].append(r)
        else:
            loose.append(r)

    chains: list[ContractChain] = []
    for key in sorted(set(roots) | set(members)):
        issues: list[Issue] = []
        candidates = sorted(roots.get(key, []), key=lambda r: (r.is_draft, r.doc_id))
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
                target = DOC_NUMBER_RE.search(op.target)
                if op.op == OpType.INVALIDATE_DOCUMENT and target and target.group(0) in by_doc:
                    by_doc[target.group(0)].invalidated_by = r.doc_id
                    issues.append(Issue(code="invalidated",
                                        message=f"{target.group(0)} признан недействительным документом {_value(r.number)}",
                                        sources=[s for s in [op.source] if s]))
                elif target and target.group(0) not in by_number:
                    issues.append(Issue(code="target_missing",
                                        message=f"{_value(r.number)} изменяет {target.group(0)}, которого нет в пакете",
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
                                    main_date=_main_date(main, sups, issues), docs=docs, issues=issues))

    for r in loose:
        chains.append(ContractChain(contract_number=_value(r.number) or r.doc_id, main_doc_id=None, docs=[
            ChainDoc(doc_id=r.doc_id, number=_value(r.number), role="unknown", order_key=_order(r)[0],
                     order_basis=_order(r)[1])],
            issues=[Issue(code="unlinked", message=f"{r.file_name}: не удалось связать с договором"), *r.issues]))
    return chains

