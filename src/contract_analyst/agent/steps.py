"""Work steps shared by the agent tools and the fixed pipeline. Each returns a compact ToolResult."""

from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date

from contract_analyst.agent.context import RunContext
from contract_analyst.analyze.cluster import cluster_contracts, summarize_cluster
from contract_analyst.assemble.chains import build_chains
from contract_analyst.assemble.consolidate import consolidate
from contract_analyst.ingest.inventory import list_inputs
from contract_analyst.ingest.reader import _error, read_document, rerecognize_pages
from contract_analyst.kb.store import KnowledgeBase
from contract_analyst.llm.ollama_client import TRANSPORT_ERRORS
from contract_analyst.models import (
    Decision,
    DocumentRecord,
    DocumentText,
    InputFile,
    Issue,
    PageMethod,
    ToolResult,
    ValueStatus,
)
from contract_analyst.ocr import get_ocr_engine
from contract_analyst.parsing.header import parse_header
from contract_analyst.report.render import ReportContext, render_report
from contract_analyst.report.verify import verify_claims
from contract_analyst.understand.understand import prompt_versions, understand_document

DATA_LIMIT = 2000


def fit(data: dict, limit: int = DATA_LIMIT) -> dict:
    """Shrink the longest lists until the JSON fits the limit, so tool results never flood the context."""
    data = json.loads(json.dumps(data, ensure_ascii=False, default=str))
    while len(json.dumps(data, ensure_ascii=False)) > limit:
        lists = [(k, v) for k, v in data.items() if isinstance(v, list) and v]
        if not lists:
            for k, v in data.items():
                if isinstance(v, str) and len(v) > 200:
                    data[k] = v[:200] + "…"
            break
        _key, longest = max(lists, key=lambda kv: len(json.dumps(kv[1], ensure_ascii=False)))
        longest.pop()
        data["truncated"] = True
    return data


def doc_line(ctx: RunContext, doc: DocumentText) -> dict:
    header = ctx.headers.get(doc.doc_id)
    methods = Counter(p.method.value for p in doc.pages)
    low = [p.number for p in doc.pages if p.method == PageMethod.OCR
           and (p.quality.ocr_mean_conf or 0) < ctx.settings.ocr_word_conf_threshold]
    return {"doc_id": doc.doc_id, "file": doc.file.name, "pages": len(doc.pages), "methods": dict(methods),
            "status": doc.status, "low_conf_pages": low[:10], "title": header.title_kind if header else None,
            "number": header.number if header else None, "parent": header.parent_number if header else None}


def _unprocessed(ctx: RunContext, file: str, reason: str) -> None:
    ctx.unprocessed.append({"file": file, "reason": reason})
    ctx.trace.event("error", "ingest", file=file, message=reason)


def _register(ctx: RunContext, doc: DocumentText) -> None:
    if doc.status == "error":
        _unprocessed(ctx, doc.file.name, ", ".join(doc.issues) or "error")
    elif doc.status == "empty":
        _unprocessed(ctx, doc.file.name, "empty: текст не найден")
    else:
        ctx.documents[doc.doc_id] = doc
        ctx.headers[doc.doc_id] = parse_header(doc.pages[0].text if doc.pages else "")
        ctx.changed("documents")


def resolve_doc_id(ctx: RunContext, ref: str) -> str | None:
    """Accept a doc_id or a file name: small models often pass the name they saw in the inventory."""
    if ref in ctx.documents:
        return ref
    for d in ctx.documents.values():
        header = ctx.headers.get(d.doc_id)
        record = ctx.records.get(d.doc_id)
        if ref in (d.file.name, d.file.name.rsplit(".", 1)[0]) or (header and header.number == ref) or (
                record and record.number.value == ref):
            return d.doc_id
    return None


def _read(ctx: RunContext, file: InputFile) -> DocumentText:
    """One file that cannot be read (OCR engine missing, page rendering failed) must not stop the run."""
    try:
        return read_document(file, ctx.ocr_engine, ctx.settings, ctx.read_cache)
    except Exception as exc:  # noqa: BLE001 — recorded as an unread file with its reason
        return _error(file, f"read_failed: {type(exc).__name__}: {exc}"[:300])


def _llm_failure(exc: Exception) -> str:
    kind = "llm_timeout" if isinstance(exc, TimeoutError) or "Timeout" in type(exc).__name__ else "llm_error"
    return f"{kind}: {type(exc).__name__}: {exc}"[:300]


def ingest_all(ctx: RunContext) -> ToolResult:
    ctx.inventory = list_inputs(ctx.input_dir)
    if ctx.only_paths is not None:
        ctx.inventory = [f for f in ctx.inventory if f.path in ctx.only_paths]
    names = {f.path: f.name for f in ctx.inventory}
    todo = []
    for f in ctx.inventory:
        if f.duplicate_of:
            _unprocessed(ctx, f.name, f"дубликат файла {names.get(f.duplicate_of, f.duplicate_of)}")
        else:
            todo.append(f)
    with ThreadPoolExecutor(max_workers=4) as pool:
        docs = list(pool.map(lambda f: _read(ctx, f), todo))
    for doc in docs:
        _register(ctx, doc)
    return ToolResult(status="ok", data=fit({
        "files": len(ctx.inventory), "read": len(ctx.documents), "unprocessed": ctx.unprocessed,
        "documents": [doc_line(ctx, d) for d in ctx.documents.values()]}))


def list_only(ctx: RunContext) -> ToolResult:
    """Variant C: the agent sees the file list and decides how to read each file."""
    ctx.inventory = list_inputs(ctx.input_dir)
    return ToolResult(status="ok", data=fit({"files": [{"file": f.name, "format": f.format.value, "bytes": f.size_bytes,
                                                       "duplicate_of": f.duplicate_of} for f in ctx.inventory]}))


def read_one(ctx: RunContext, file_name: str) -> ToolResult:
    if not ctx.inventory:
        ctx.inventory = list_inputs(ctx.input_dir)
    match = next((f for f in ctx.inventory if f.name == file_name), None)
    if match is None:
        return ToolResult(status="error", error_type="not_found", hint="Нет такого файла: вызови list_inputs")
    doc = _read(ctx, match)
    _register(ctx, doc)
    line = doc_line(ctx, doc) if doc.status not in ("error", "empty") else {"file": doc.file.name, "issues": doc.issues}
    if doc.status == "low_quality":
        return ToolResult(status="low_quality", data=fit(line), hint=(
            "Есть страницы с низкой уверенностью OCR: вызови rerecognize(doc_id, pages, strategy) "
            "или откажись от значений с этих страниц"))
    if doc.status in ("error", "empty"):
        return ToolResult(status="error" if doc.status == "error" else "empty", data=fit(line),
                          error_type=doc.issues[0] if doc.issues else doc.status,
                          hint="Документ не прочитан: он попадёт в раздел «Не обработано»")
    return ToolResult(status="ok", data=fit(line))


def rerecognize(ctx: RunContext, doc_id: str, pages: list[int], strategy: str) -> ToolResult:
    doc_id = resolve_doc_id(ctx, doc_id) or doc_id
    doc = ctx.documents.get(doc_id)
    if doc is None:
        return ToolResult(status="error", error_type="not_found", hint="Нет такого doc_id")
    engine, mode, dpi = ctx.ocr_engine, ctx.settings.ocr_preprocess, ctx.settings.ocr_dpi
    if strategy == "higher_dpi":
        dpi = 400
    elif strategy in ("preprocess", "rotate"):
        mode = "deskew_sauvola"
    elif strategy == "alt_engine":
        engine = get_ocr_engine("rapidocr" if ctx.ocr_engine.name == "tesseract" else "tesseract")
    pages = [p for p in pages if 1 <= p <= len(doc.pages)]
    better = rerecognize_pages(doc, pages, engine, mode, dpi, ctx.settings)
    report, improved = [], False
    for p in pages:
        before = doc.pages[p - 1].quality.ocr_mean_conf or 0.0
        after = better.pages[p - 1].quality.ocr_mean_conf or 0.0
        report.append({"page": p, "before": round(before, 3), "after": round(after, 3)})
        improved = improved or after > before
    if improved:
        ctx.documents[doc_id] = better
        ctx.changed("documents")
    status = "ok" if (better if improved else doc).status == "ok" else "low_quality"
    return ToolResult(status=status, data=fit({"doc_id": doc_id, "strategy": strategy, "pages": report,
                                               "adopted": improved}),
                      hint=None if status == "ok" else "Качество всё ещё низкое: значения с этих страниц будут отклонены шлюзом")


def _merge_records(old: DocumentRecord, new: DocumentRecord, groups: list[str]) -> DocumentRecord:
    update = {}
    if "header" in groups:
        update.update({f: getattr(new, f) for f in ("kind", "number", "doc_date", "city", "parent", "effective_from",
                                                     "subject_category", "parties", "title", "is_draft")})
    if {"core_terms", "dealer"} & set(groups):
        update["terms"] = {**old.terms, **new.terms}
    if "money" in groups:
        update["rates"] = new.rates
    if "ops" in groups:
        update["ops"] = new.ops
    update["issues"] = [*old.issues, *new.issues]
    return old.model_copy(update=update)


def _counts(record: DocumentRecord) -> dict:
    values = [record.number, record.doc_date, record.parent, record.effective_from, *record.terms.values()]
    statuses = Counter(v.status.value for v in values) + Counter(r.status.value for r in [*record.rates, *record.ops])
    return dict(statuses)


def analyze_docs(ctx: RunContext, doc_ids: list[str], groups: list[str] | None = None) -> ToolResult:
    done, skipped = [], []
    for ref in doc_ids:
        doc_id = resolve_doc_id(ctx, ref) or ref
        doc = ctx.documents.get(doc_id)
        if doc is None or (ctx.only_doc_ids is not None and doc_id not in ctx.only_doc_ids):
            skipped.append(doc_id)
            continue
        try:
            record = understand_document(doc, ctx.headers[doc_id], ctx.client, ctx.settings, groups)
        except TRANSPORT_ERRORS as exc:
            _unprocessed(ctx, doc.file.name, _llm_failure(exc))
            skipped.append(doc_id)
            continue
        for issue in record.issues:
            if issue.code in ("llm_invalid_json", "llm_timeout", "llm_truncated"):
                ctx.trace.event("error", "tools", doc_id=doc_id, message=f"{issue.code}: {issue.message}")
        if groups and doc_id in ctx.records:
            record = _merge_records(ctx.records[doc_id], record, groups)
        if ctx.records.get(doc_id) != record:
            ctx.records[doc_id] = record
            ctx.changed("records")
        rejected = [r for r in [*record.rates, *record.ops] if r.status == ValueStatus.REJECTED]
        if rejected:
            ctx.trace.event("gate_reject", "tools", doc_id=doc_id, count=len(rejected),
                            reasons=[r.reason for r in rejected][:5])
        done.append({"doc_id": doc_id, "number": record.number.value, "kind": record.kind.value,
                     "parent": record.parent.value.number if record.parent.value else None, "values": _counts(record),
                     "issues": [i.code for i in record.issues]})
    status = "ok" if done else "empty"
    return ToolResult(status=status, data=fit({"analyzed": done, "skipped": skipped}),
                      hint=None if done else "Ни один документ не разобран: проверь doc_id через list_inputs")


def assemble_chains(ctx: RunContext) -> ToolResult:
    chains = build_chains(list(ctx.records.values()))
    ctx.built("chains")
    if chains != ctx.chains:
        ctx.chains = chains
        ctx.changed("chains")
    return ToolResult(status="ok" if ctx.chains else "empty", data=fit({"chains": [
        {"contract": c.contract_number, "documents": len(c.docs), "has_main": c.main_doc_id is not None,
         "issues": sorted({i.code for i in c.issues})} for c in ctx.chains]}))


def consolidate_all(ctx: RunContext, contract_number: str | None = None, as_of: date | None = None) -> ToolResult:
    when = as_of or ctx.as_of
    out, results = [], {}
    for chain in ctx.chains:
        if contract_number and chain.contract_number != contract_number:
            continue
        if ctx.settings.consolidation_mode == "llm":
            from contract_analyst.assemble.merge_llm import consolidate_llm

            try:
                result = consolidate_llm(chain, ctx.records, when, ctx.client)
            except TRANSPORT_ERRORS as exc:
                result = consolidate(chain, ctx.records, when)
                result.issues.append(Issue(code="llm_merge_failed", message=f"{_llm_failure(exc)}: свёрнуто кодом"))
        else:
            result = consolidate(chain, ctx.records, when)
        results[chain.contract_number] = result
        out.append({"contract": chain.contract_number, "status": result.status, "since": result.status_since,
                    "terms": len(result.terms), "rates": len(result.rates), "planned_rates": len(result.planned_rates)})
    if when == ctx.as_of:
        # A full run replaces the set, so contracts whose chain is gone do not linger.
        merged = {**ctx.consolidated, **results} if contract_number else results
        if merged != ctx.consolidated:
            ctx.consolidated = merged
            ctx.changed("consolidated")
        if not contract_number:
            ctx.built("consolidated")
    return ToolResult(status="ok" if out else "empty", data=fit({"as_of": when, "contracts": out}))


def cluster_all(ctx: RunContext) -> ToolResult:
    clusters = cluster_contracts(list(ctx.consolidated.values()), ctx.records, ctx.chains, ctx.settings)
    for cluster in clusters:
        try:
            cluster.summary = summarize_cluster(cluster, ctx.consolidated, ctx.client)
        except TRANSPORT_ERRORS as exc:
            ctx.trace.event("error", "tools", message=_llm_failure(exc))
            cluster.summary = None
    ctx.built("clusters")
    if clusters != ctx.clusters:
        ctx.clusters = clusters
        ctx.changed("clusters")
    return ToolResult(status="ok" if ctx.clusters else "empty", data=fit({"clusters": [
        {"label": c.label, "members": c.members, "differences": len(c.differences)} for c in ctx.clusters]}))


def record_decision(ctx: RunContext, decision: Decision) -> ToolResult:
    ctx.decisions.append(decision)
    ctx.changed("decisions")
    return ToolResult(status="ok", data={"decisions": len(ctx.decisions)})


def write_report(ctx: RunContext) -> ToolResult:
    valid = {f"doc:{d.doc_id}:p{p.number}" for d in ctx.documents.values() for p in d.pages}
    report_ctx = ReportContext(
        run_id=ctx.run_dir.name, as_of=ctx.as_of, model=ctx.client.model_name, prompt_versions=prompt_versions(),
        ocr=f"{ctx.ocr_engine.name}/{ctx.settings.ocr_preprocess}", documents=list(ctx.documents.values()),
        records=ctx.records, chains=ctx.chains, consolidated=ctx.consolidated, clusters=ctx.clusters,
        decisions=ctx.decisions, reflection=ctx.reflection, unprocessed=ctx.unprocessed,
        trace_path=str(ctx.trace.path), metrics_path=str(ctx.run_dir / "metrics.json"))
    markdown, removed = verify_claims(render_report(report_ctx), valid)
    ctx.report_path = ctx.run_dir / "report.md"
    ctx.report_path.write_text(markdown, encoding="utf-8")
    ctx.removed_claims = removed
    KnowledgeBase(ctx.run_dir / "kb").save(documents=ctx.documents, records=ctx.records, chains=ctx.chains,
                                           consolidated=ctx.consolidated, clusters=ctx.clusters, decisions=ctx.decisions)
    # Run-independent: the result goes back to the model, and a timestamped path would change the next turn's
    # cache key, so a recorded run could never be replayed.
    return ToolResult(status="ok", data={"report": ctx.report_path.name, "removed_claims": removed})
