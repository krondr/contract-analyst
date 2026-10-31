"""Golden set tooling: prefill from deterministic parsing, text dumps for annotators."""

from __future__ import annotations

from pathlib import Path

import cv2

from contract_analyst.eval.golden import GoldenDocument, GoldenSet, GoldenValue
from contract_analyst.models import DocKind, DocumentText, PageMethod
from contract_analyst.ocr.render import render_page
from contract_analyst.parsing.header import detect_draft, parse_header

_KIND = {"contract": DocKind.MAIN, "supplement": DocKind.SUPPLEMENT, "agreement": DocKind.FRAMEWORK,
         "unknown": DocKind.UNKNOWN}


def _rel(doc: DocumentText, root: Path | None) -> str:
    if root is None:
        return doc.file.name
    try:
        return str(Path(doc.file.path).relative_to(root))
    except ValueError:
        return doc.file.name


def _gv(value, quote: str | None) -> GoldenValue:
    if value is None:
        return GoldenValue(status="absent")
    text = value.isoformat() if hasattr(value, "isoformat") else str(value)
    return GoldenValue(value=text, page=1, quote=quote)


def prefill(docs: list[DocumentText], root: Path | None = None) -> GoldenSet:
    documents = []
    for doc in docs:
        first = doc.pages[0].text if doc.pages else ""
        h = parse_header(first)
        documents.append(GoldenDocument(
            file=_rel(doc, root), doc_id=doc.doc_id, kind=_KIND[h.title_kind], is_draft=detect_draft(doc.file.name, first),
            number=_gv(h.number, h.spans.get("number")), doc_date=_gv(h.doc_date, h.spans.get("doc_date")),
            city=_gv(h.city, h.spans.get("city")), parent_number=_gv(h.parent_number, h.spans.get("parent")),
            parent_date=_gv(h.parent_date, h.spans.get("parent"))))
    return GoldenSet(documents=documents)


def dump_texts(docs: list[DocumentText], out_dir: Path, png_dpi: int = 150) -> list[Path]:
    """One text file per document with page markers; OCR pages also as PNG for visual checks."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for doc in docs:
        lines = [f"# file: {doc.file.name}", f"# doc_id: {doc.doc_id}", f"# status: {doc.status}"]
        for p in doc.pages:
            conf = f", ocr_conf={p.quality.ocr_mean_conf:.2f}" if p.quality.ocr_mean_conf is not None else ""
            lines.append(f"\n=== PAGE {p.number} ({p.method.value}{conf}) ===\n{p.text}")
            if p.method == PageMethod.OCR and Path(doc.file.path).suffix.lower() == ".pdf":
                png = out_dir / "png" / f"{doc.doc_id}_p{p.number}.png"
                png.parent.mkdir(exist_ok=True)
                image = render_page(Path(doc.file.path), p.number, png_dpi)
                cv2.imwrite(str(png), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        path = out_dir / f"{doc.doc_id}.txt"
        path.write_text("\n".join(lines), encoding="utf-8")
        written.append(path)
    return written


# --- diff, consistency check, review page -------------------------------------------------------------

import html as _html
import json as _json
import random as _random
import re as _re

from pydantic import BaseModel as _BaseModel

from contract_analyst.eval.golden import GoldenRate
from contract_analyst.eval.text_metrics import cer

_SCALARS = ("number", "doc_date", "city", "parent_number", "parent_date", "effective_from", "counterparty_type")
_WORD = _re.compile(r"[0-9a-zа-яё]{3,}")


class GoldenDiff(_BaseModel):
    path: str
    a: str | None
    b: str | None


def _words(text: str | None) -> set[str]:
    return set(_WORD.findall((text or "").casefold().replace("ё", "е")))


def _similar(a: str | None, b: str | None) -> bool:
    wa, wb = _words(a), _words(b)
    if not wa or not wb:
        return (a or "").strip() == (b or "").strip()
    return len(wa & wb) / min(len(wa), len(wb)) >= 0.6


def _rate_key(rates: list[GoldenRate]) -> set[tuple]:
    return {(r.amount.normalize(), r.valid_from, r.valid_to) for r in rates}


def _value_str(v: GoldenValue) -> str:
    return f"{v.value} ({v.status})"


def diff_golden(a: GoldenSet, b: GoldenSet) -> list[GoldenDiff]:
    diffs: list[GoldenDiff] = []

    def add(path: str, x, y) -> None:
        diffs.append(GoldenDiff(path=path, a=None if x is None else str(x), b=None if y is None else str(y)))

    docs_a, docs_b = {d.doc_id: d for d in a.documents}, {d.doc_id: d for d in b.documents}
    for doc_id in sorted(set(docs_a) | set(docs_b)):
        da, db = docs_a.get(doc_id), docs_b.get(doc_id)
        if da is None or db is None:
            add(f"documents[{doc_id}]", da and "есть", db and "есть")
            continue
        for field in _SCALARS:
            va, vb = getattr(da, field), getattr(db, field)
            if (va.value or "").strip() != (vb.value or "").strip() or va.status != vb.status:
                add(f"documents[{doc_id}].{field}", _value_str(va), _value_str(vb))
        for field in ("kind", "is_draft", "subject_category"):
            if getattr(da, field) != getattr(db, field):
                add(f"documents[{doc_id}].{field}", getattr(da, field), getattr(db, field))
        for key in sorted(set(da.terms) | set(db.terms)):
            ta, tb = da.terms.get(key), db.terms.get(key)
            if ta is None or tb is None or not _similar(ta.value, tb.value):
                add(f"documents[{doc_id}].terms.{key}", ta and ta.value, tb and tb.value)
        if _rate_key(da.rates) != _rate_key(db.rates):
            add(f"documents[{doc_id}].rates", sorted(map(str, _rate_key(da.rates))), sorted(map(str, _rate_key(db.rates))))
        ops_a = {(o.op, (o.target or "").replace(" ", ""), o.effective_from) for o in da.ops}
        ops_b = {(o.op, (o.target or "").replace(" ", ""), o.effective_from) for o in db.ops}
        if ops_a != ops_b:
            add(f"documents[{doc_id}].ops", sorted(map(str, ops_a)), sorted(map(str, ops_b)))

    chains_a, chains_b = {c.contract_number: c for c in a.chains}, {c.contract_number: c for c in b.chains}
    for number in sorted(set(chains_a) | set(chains_b)):
        ca, cb = chains_a.get(number), chains_b.get(number)
        if ca is None or cb is None:
            add(f"chains[{number}]", ca and "есть", cb and "есть")
            continue
        for field in ("main_doc_id", "order", "invalidated"):
            if getattr(ca, field) != getattr(cb, field):
                add(f"chains[{number}].{field}", getattr(ca, field), getattr(cb, field))
        for when in sorted(set(ca.status_at) | set(cb.status_at)):
            if ca.status_at.get(when) != cb.status_at.get(when):
                add(f"chains[{number}].status_at[{when}]", ca.status_at.get(when), cb.status_at.get(when))
        for when in sorted(set(ca.rates_at) | set(cb.rates_at)):
            if _rate_key(ca.rates_at.get(when, [])) != _rate_key(cb.rates_at.get(when, [])):
                add(f"chains[{number}].rates_at[{when}]", ca.rates_at.get(when), cb.rates_at.get(when))
        for when in sorted(set(ca.terms_at) | set(cb.terms_at)):
            ta, tb = ca.terms_at.get(when, {}), cb.terms_at.get(when, {})
            for key in sorted(set(ta) | set(tb)):
                if not _similar(ta.get(key), tb.get(key)):
                    add(f"chains[{number}].terms_at[{when}].{key}", ta.get(key), tb.get(key))

    clusters_a = {(c.category, frozenset(c.members)) for c in a.clusters}
    clusters_b = {(c.category, frozenset(c.members)) for c in b.clusters}
    if clusters_a != clusters_b:
        add("clusters", sorted(map(str, clusters_a)), sorted(map(str, clusters_b)))
    traps_a, traps_b = {t.trap_id for t in a.traps}, {t.trap_id for t in b.traps}
    if traps_a != traps_b:
        add("traps", sorted(traps_a), sorted(traps_b))
    scans_a, scans_b = {(s.file, s.page): s.text for s in a.scans}, {(s.file, s.page): s.text for s in b.scans}
    for key in sorted(set(scans_a) | set(scans_b)):
        ta, tb = scans_a.get(key), scans_b.get(key)
        if ta is None or tb is None or cer(ta, tb) > 0.05:
            add(f"scans[{key[0]}:{key[1]}]", ta, tb)
    return diffs


def _to_record(doc: GoldenDocument):
    from contract_analyst.models import (
        AmendmentOp,
        DocumentRecord,
        Extracted,
        ParentRef,
        RateRow,
        Source,
        ValueStatus,
    )

    def ext(v: GoldenValue):
        if v.status != "accepted" or v.value is None:
            return Extracted(status=v.status if v.status != "accepted" else ValueStatus.ABSENT)
        return Extracted(value=v.value, status=ValueStatus.ACCEPTED,
                         source=Source(doc_id=doc.doc_id, page=v.page or 1, quote=v.quote or ""))

    parent = (Extracted(value=ParentRef(number=doc.parent_number.value,
                                        date_stated=doc.parent_date.value if doc.parent_date.status == "accepted" else None),
                        status=ValueStatus.ACCEPTED)
              if doc.parent_number.status == "accepted" and doc.parent_number.value else Extracted(status=ValueStatus.ABSENT))
    return DocumentRecord(
        doc_id=doc.doc_id, file_name=doc.file, title=doc.file, is_draft=doc.is_draft,
        kind=Extracted(value=doc.kind, status=ValueStatus.ACCEPTED), number=ext(doc.number).model_dump(),
        doc_date=ext(doc.doc_date).model_dump(), city=ext(doc.city).model_dump(), parent=parent.model_dump(),
        effective_from=ext(doc.effective_from).model_dump(), effective_to=ext(doc.effective_to).model_dump(),
        subject_category=Extracted(value=doc.subject_category, status=ValueStatus.ACCEPTED),
        terms={k: ext(v) for k, v in doc.terms.items()},
        rates=[RateRow(label=r.label, amount=r.amount, vat=r.vat, valid_from=r.valid_from, valid_to=r.valid_to,
                       status=ValueStatus.ACCEPTED) for r in doc.rates],
        ops=[AmendmentOp(op=o.op, target=o.target, effective_from=o.effective_from, status=ValueStatus.ACCEPTED)
             for o in doc.ops])


def _check_value(doc: GoldenDocument, field: str, v: GoldenValue, text_doc: DocumentText, errors: list[str]) -> None:
    from contract_analyst.gate.checks import check_format, check_quote_present, check_value_in_quote

    if v.status != "accepted":
        return
    where = f"documents[{doc.doc_id}].{field}"
    if v.value is None or v.page is None or not v.quote:
        errors.append(f"{where}: accepted value needs value, page and quote")
        return
    if not 1 <= v.page <= len(text_doc.pages):
        errors.append(f"{where}: page {v.page} does not exist")
        return
    if not check_quote_present(v.quote, text_doc.pages[v.page - 1], 85).ok:
        errors.append(f"{where}: quote not found on page {v.page}")
        return
    kind = {"number": "doc_number", "parent_number": "doc_number", "doc_date": "date", "parent_date": "date",
            "effective_from": "date"}.get(field)
    if kind and not (check_format(v.value, kind).ok and check_value_in_quote(v.value, v.quote, kind).ok):
        errors.append(f"{where}: value {v.value} does not follow from its quote or has a bad format")


def check_golden(gs: GoldenSet, docs: dict[str, DocumentText], required_traps: set[str] | None = None) -> list[str]:
    from contract_analyst.assemble.chains import build_chains
    from contract_analyst.assemble.consolidate import consolidate
    from contract_analyst.gate.checks import check_quote_present

    errors: list[str] = []
    for doc in gs.documents:
        text_doc = docs.get(doc.doc_id)
        if text_doc is None:
            errors.append(f"documents[{doc.doc_id}]: no such document in data")
            continue
        for field in _SCALARS:
            _check_value(doc, field, getattr(doc, field), text_doc, errors)
        for key, value in doc.terms.items():
            _check_value(doc, f"terms.{key}", value, text_doc, errors)
        for i, rate in enumerate(doc.rates):
            where = f"documents[{doc.doc_id}].rates[{i}]"
            if rate.page and rate.quote and 1 <= rate.page <= len(text_doc.pages):
                if not check_quote_present(rate.quote, text_doc.pages[rate.page - 1], 85).ok:
                    errors.append(f"{where}: quote not found on page {rate.page}")
            else:
                errors.append(f"{where}: rate needs page and quote")

    known = {d.doc_id for d in gs.documents}
    for chain in gs.chains:
        if set(chain.order) - known or set(chain.invalidated) - set(chain.order):
            errors.append(f"chains[{chain.contract_number}]: order/invalidated reference unknown documents")
        if chain.main_doc_id and chain.main_doc_id not in chain.order:
            errors.append(f"chains[{chain.contract_number}]: main_doc_id not in order")

    records = [_to_record(d) for d in gs.documents]
    by_id = {r.doc_id: r for r in records}
    built = {c.contract_number: c for c in build_chains(records)}
    for chain in gs.chains:
        assembled = built.get(chain.contract_number)
        if assembled is None:
            errors.append(f"chains[{chain.contract_number}]: documents do not assemble into this chain")
            continue
        for when in gs.as_of_dates:
            result = consolidate(assembled, by_id, when)
            if when in chain.status_at and chain.status_at[when] != result.status:
                errors.append(f"chains[{chain.contract_number}].status_at[{when}]: golden {chain.status_at[when]}, "
                              f"recomputed {result.status}")
            if when in chain.rates_at:
                recomputed = {(r.amount.normalize(), r.valid_from, r.valid_to) for r in result.rates}
                if _rate_key(chain.rates_at[when]) != recomputed:
                    errors.append(f"chains[{chain.contract_number}].rates_at[{when}]: golden differs from recomputed")
            for key, value in chain.terms_at.get(when, {}).items():
                state = result.terms.get(key)
                if state is None or not _similar(value, state.current.value):
                    errors.append(f"chains[{chain.contract_number}].terms_at[{when}].{key}: golden differs from recomputed")

    if required_traps:
        present = {t.trap_id for t in gs.traps}
        errors.extend(f"missing trap {t}" for t in sorted(required_traps - present, key=lambda x: int(x[1:])))
    chain_numbers = {c.contract_number for c in gs.chains}
    for cluster in gs.clusters:
        if set(cluster.members) - chain_numbers:
            errors.append(f"clusters[{cluster.category}]: members without chains")
    return errors


_PAGE_STYLE = """
body{font-family:-apple-system,Segoe UI,sans-serif;margin:16px;color:#1f2328;background:#fff}
table{border-collapse:collapse;width:100%}td,th{border:1px solid #d0d7de;padding:6px;vertical-align:top;font-size:14px}
tr.disputed{background:#fff8c5}td.q{font-family:ui-monospace,monospace;font-size:13px}
textarea{width:100%;height:160px}button{margin:12px 0;padding:8px 14px}
"""


def build_review_page(gs: GoldenSet, disputed: list[GoldenDiff], out_html: Path, sample_share: float = 0.2,
                      seed: int = 0) -> Path:
    rng = _random.Random(seed)
    rows: list[str] = []
    esc = _html.escape
    for i, d in enumerate(disputed):
        rows.append(f'<tr class="disputed" data-id="d{i}"><td>{esc(d.path)}</td><td>А: {esc(d.a or "—")}<br>'
                    f'Б: {esc(d.b or "—")}</td><td></td><td><label><input type="radio" name="d{i}" value="a">А</label> '
                    f'<label><input type="radio" name="d{i}" value="b">Б</label> <label><input type="radio" name="d{i}" '
                    f'value="other">другое</label><input type="text" placeholder="комментарий"></td></tr>')
    for doc in gs.documents:
        values = [(f, getattr(doc, f)) for f in _SCALARS if getattr(doc, f).status == "accepted"]
        values += [(f"terms.{k}", v) for k, v in doc.terms.items() if v.status == "accepted"]
        if not values:
            continue
        k = max(1, round(len(values) * sample_share))
        for field, v in rng.sample(values, min(k, len(values))):
            rid = f"s-{doc.doc_id}-{field}"
            rows.append(f'<tr class="sample" data-id="{esc(rid)}"><td>{esc(doc.file)}<br>{esc(field)}</td>'
                        f'<td>{esc(v.value or "")}</td><td class="q">стр. {v.page}: {esc(v.quote or "")}</td>'
                        f'<td><label><input type="radio" name="{esc(rid)}" value="ok">✓</label> '
                        f'<label><input type="radio" name="{esc(rid)}" value="bad">✗</label>'
                        f'<input type="text" placeholder="комментарий"></td></tr>')
    script = ("document.getElementById('collect').onclick=()=>{const r=[];document.querySelectorAll('tr[data-id]')"
              ".forEach(tr=>{const c=tr.querySelector('input[type=radio]:checked');const t=tr.querySelector("
              "'input[type=text]');r.push({id:tr.dataset.id,mark:c?c.value:null,comment:t?t.value:''})});"
              "document.getElementById('out').value=JSON.stringify(r,null,1)};")
    page = (f"<!doctype html><html lang='ru'><head><meta charset='utf-8'><title>Проверка эталона</title>"
            f"<style>{_PAGE_STYLE}</style></head><body><h1>Проверка эталона</h1>"
            f"<p>Жёлтые строки — расхождения разметчиков: выберите верный вариант. Остальные — случайная выборка "
            f"{int(sample_share * 100)}%: сверьте значение с цитатой.</p><table><tr><th>Поле</th><th>Значение</th>"
            f"<th>Цитата</th><th>Отметка</th></tr>{''.join(rows)}</table><button id='collect'>Собрать отметки</button>"
            f"<textarea id='out' readonly></textarea><script>{script}</script></body></html>")
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(page, encoding="utf-8")
    return out_html


def diffs_to_json(diffs: list[GoldenDiff]) -> str:
    return _json.dumps([d.model_dump() for d in diffs], ensure_ascii=False, indent=1)


__all__ = ["GoldenDiff", "build_review_page", "check_golden", "diff_golden", "diffs_to_json", "dump_texts", "prefill"]
