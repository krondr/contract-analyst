"""State revisions: repeated calls are blocked only while the state they read is unchanged; stale results are redone."""

import json
from datetime import date

import pytest

from contract_analyst.agent import steps
from contract_analyst.agent.guards import Guard
from contract_analyst.agent.runner import run_analysis
from contract_analyst.agent.tools import dispatch
from contract_analyst.agent.trace import read_trace
from contract_analyst.config import load_settings
from contract_analyst.llm.client import AssistantTurn, ToolCall
from contract_analyst.llm.scripted import ScriptedClient
from tests.agent.conftest import make_ctx
from tests.fixtures.builders import FakeOcrEngine, make_text_pdf
from tests.fixtures.builders import make_record as rec

MAIN = "D990000001-01"
RECORDS = {
    "Договор.pdf": {"number": MAIN, "kind": "main", "doc_date": date(2024, 1, 10), "rates": [("Аренда", 100000, None, None)]},
    "ДС1.pdf": {"number": "D990000002-01", "parent": MAIN, "effective": date(2025, 4, 1),
                "rates": [("Аренда", 120000, None, None)]},
    "ДС2.pdf": {"number": "D990000003-01", "parent": MAIN, "effective": date(2026, 1, 1),
                "rates": [("Аренда", 150000, None, None)]},
}
DONE = ['{"summary": "Аренда."}', AssistantTurn(content="Готово"), '{"goal_met": true, "gaps": []}']


@pytest.fixture
def package(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    make_text_pdf(pkg / "Договор.pdf", [f"Договор № {MAIN}\nг. Тверь 10.01.2024 г.\n1. Предмет."])
    make_text_pdf(pkg / "ДС1.pdf", [f"Дополнительное соглашение № D990000002-01\nк Договору № {MAIN} от 10.01.2024 г."])
    make_text_pdf(pkg / "ДС2.pdf", [f"Дополнительное соглашение № D990000003-01\nк Договору № {MAIN} от 10.01.2024 г."])
    return pkg


@pytest.fixture(autouse=True)
def stub_understanding(monkeypatch):
    def fake(doc, header, client, settings, groups=None):
        spec = dict(RECORDS[doc.file.name])
        return rec(doc.doc_id, spec.pop("number"), file_name=doc.file.name, **spec)

    monkeypatch.setattr(steps, "understand_document", fake)


def chat(*calls, content=""):
    return AssistantTurn(content=content, tool_calls=[ToolCall(id=f"{n}{i}", name=n, args={"thought": "шаг", **a})
                                                      for i, (n, a) in enumerate(calls)])


def run(package, script, tmp_path):
    return run_analysis(package, load_settings(), date(2026, 10, 30), client=ScriptedClient(script),
                        ocr_engine=FakeOcrEngine(), out_dir=tmp_path / "runs", read_cache=None, console=False)


def current_terms(report_path):
    return report_path.read_text(encoding="utf-8").split("#### Действующие условия")[1].split("####")[0]


def test_chains_and_consolidation_rerun_after_a_new_document(package, tmp_path):
    script = ['{"items": ["План"]}',
              chat(("analyze_document", {"doc_id": "Договор.pdf"}), ("analyze_document", {"doc_id": "ДС1.pdf"})),
              chat(("build_chains", {}), ("consolidate", {})),
              chat(("analyze_document", {"doc_id": "ДС2.pdf"})),
              chat(("build_chains", {}), ("consolidate", {}), ("cluster_and_compare", {})), *DONE]
    r = run(package, script, tmp_path)
    trace = read_trace(r.trace_path)
    assert not [e for e in trace if e["event"] == "tool_result" and e["error_type"] == "repeat_call"]
    assert "150 000" in current_terms(r.report_path)
    assert "ДС2.pdf" in r.report_path.read_text(encoding="utf-8").split("## 4.")[1]
    assert json.loads((r.run_dir / "metrics.json").read_text())["safety_net"] == []


def test_safety_net_cascades_when_the_agent_leaves_chains_stale(package, tmp_path):
    script = ['{"items": ["План"]}',
              chat(("analyze_document", {"doc_id": "Договор.pdf"}), ("analyze_document", {"doc_id": "ДС1.pdf"})),
              chat(("build_chains", {}), ("consolidate", {}), ("cluster_and_compare", {})), '{"summary": "Аренда."}',
              chat(("analyze_document", {"doc_id": "ДС2.pdf"})),
              AssistantTurn(content="Готово"), '{"goal_met": true, "gaps": []}',
              AssistantTurn(content="Готово"), '{"goal_met": true, "gaps": []}', '{"summary": "Аренда."}']
    r = run(package, script, tmp_path)
    first = next(e for e in read_trace(r.trace_path) if e["event"] == "reflection")
    checks = {i["item"]: i["ok"] for i in first["items"]}
    assert checks["Цепочки договоров собраны"] is False and checks["Условия свёрнуты на дату анализа"] is False
    assert json.loads((r.run_dir / "metrics.json").read_text())["safety_net"] == [
        "assemble_chains", "consolidate_all", "cluster_all"]
    assert "150 000" in current_terms(r.report_path)


def test_repeat_is_blocked_while_the_state_it_reads_is_unchanged(tmp_path, package):
    ctx = make_ctx(tmp_path, package, client=ScriptedClient(['{"summary": "Аренда."}']))
    guard = Guard(20, 60)
    steps.ingest_all(ctx)
    steps.analyze_docs(ctx, list(ctx.documents))
    assert dispatch(ctx, chat(("build_chains", {})).tool_calls[0], guard).status == "ok"
    assert dispatch(ctx, chat(("build_chains", {})).tool_calls[0], guard).error_type == "repeat_call"
    assert dispatch(ctx, chat(("consolidate", {})).tool_calls[0], guard).status == "ok"
    assert dispatch(ctx, chat(("consolidate", {})).tool_calls[0], guard).error_type == "repeat_call"
    assert dispatch(ctx, chat(("cluster_and_compare", {})).tool_calls[0], guard).status in ("ok", "empty")


def test_clustering_before_consolidation_can_be_redone(tmp_path, package):
    ctx = make_ctx(tmp_path, package, client=ScriptedClient(['{"summary": "Аренда."}']))
    guard = Guard(20, 60)
    steps.ingest_all(ctx)
    steps.analyze_docs(ctx, list(ctx.documents))
    steps.assemble_chains(ctx)
    assert dispatch(ctx, chat(("cluster_and_compare", {})).tool_calls[0], guard).status == "empty"
    dispatch(ctx, chat(("consolidate", {})).tool_calls[0], guard)
    assert dispatch(ctx, chat(("cluster_and_compare", {})).tool_calls[0], guard).status == "ok"


def test_analysis_can_be_redone_after_rerecognize_changes_the_text(tmp_path, package, monkeypatch):
    from contract_analyst.models import PageMethod, PageQuality

    ctx = make_ctx(tmp_path, package)
    guard = Guard(20, 60)
    steps.ingest_all(ctx)
    doc_id = steps.resolve_doc_id(ctx, "ДС1.pdf")
    doc = ctx.documents[doc_id]
    scanned = doc.pages[0].model_copy(update={"method": PageMethod.OCR,
                                              "quality": PageQuality(chars=40, ocr_mean_conf=0.5, low_conf_ratio=0.6)})
    ctx.documents[doc_id] = doc.model_copy(update={"pages": [scanned], "status": "low_quality"})
    better = doc.model_copy(update={"pages": [scanned.model_copy(update={
        "quality": PageQuality(chars=40, ocr_mean_conf=0.97, low_conf_ratio=0.0)})], "status": "ok"})
    monkeypatch.setattr(steps, "rerecognize_pages", lambda *a, **k: better)
    analyze = chat(("analyze_document", {"doc_id": doc_id})).tool_calls[0]
    assert dispatch(ctx, analyze, guard).status == "ok"
    assert dispatch(ctx, analyze, guard).error_type == "repeat_call"
    redo = chat(("rerecognize", {"doc_id": doc_id, "pages": [1], "strategy": "higher_dpi"})).tool_calls[0]
    assert dispatch(ctx, redo, guard).data["adopted"] is True
    assert dispatch(ctx, analyze, guard).status == "ok"
