import json

import pytest

from contract_analyst.agent.guards import Guard
from contract_analyst.agent.steps import ingest_all
from contract_analyst.agent.tools import TOOLS, dispatch, tool_schemas
from contract_analyst.agent.trace import read_trace
from contract_analyst.llm.client import ToolCall
from tests.agent.conftest import make_ctx
from tests.fixtures.builders import FakeOcrEngine, make_text_pdf


def call(name, **args):
    return ToolCall(id="c1", name=name, args={"thought": "проверяю шаг", **args})


def test_every_tool_offers_thought_and_has_schema():
    for spec in TOOLS.values():
        assert "thought" in spec.args_model.model_json_schema()["properties"]
    names = {s["function"]["name"] for s in tool_schemas()}
    assert {"list_inputs", "read_document", "rerecognize", "analyze_document", "build_chains", "consolidate",
            "cluster_and_compare", "record_decision", "write_report"} <= names


def test_dispatch_logs_thought_then_call_then_result(ctx):
    result = dispatch(ctx, call("list_inputs"), Guard(10, 60))
    assert result.status == "ok" and result.data["files"] == 3
    events = [e["event"] for e in read_trace(ctx.trace.path)]
    assert events == ["thought", "tool_call", "tool_result"]
    assert read_trace(ctx.trace.path)[1]["thought"] == "проверяю шаг"


def test_repeat_call_blocked(ctx):
    guard = Guard(10, 60)
    dispatch(ctx, call("list_inputs"), guard)
    again = dispatch(ctx, ToolCall(id="c2", name="list_inputs", args={"thought": "другая мысль"}), guard)
    assert again.status == "error" and again.error_type == "repeat_call"


def test_step_budget_exceeded(ctx):
    guard = Guard(max_steps=1, max_seconds=60)
    dispatch(ctx, call("list_inputs"), guard)
    assert dispatch(ctx, call("build_chains"), guard).error_type == "budget_exceeded"


def test_exception_becomes_error_result(ctx, monkeypatch):
    from contract_analyst.agent import steps

    monkeypatch.setattr(steps, "build_chains", lambda records: 1 / 0)
    result = dispatch(ctx, call("build_chains"), Guard(10, 60))
    assert result.status == "error" and result.error_type == "exception"
    assert any(e["event"] == "error" for e in read_trace(ctx.trace.path))


def test_bad_args_and_unknown_tool(ctx):
    bad = ToolCall(id="x", name="rerecognize", args={"doc_id": "a", "pages": "все", "strategy": "magic"})
    assert dispatch(ctx, bad, Guard(10, 60)).error_type == "bad_args"
    assert dispatch(ctx, call("format_disk"), Guard(10, 60)).error_type == "unknown_tool"


def test_invalid_calls_count_against_the_budget(ctx):
    guard = Guard(max_steps=2, max_seconds=60)
    dispatch(ctx, ToolCall(id="x", name="analyze_document", args={}), guard)
    dispatch(ctx, call("format_disk"), guard)
    assert dispatch(ctx, call("list_inputs"), guard).error_type == "budget_exceeded"


def test_consecutive_errors_stop_the_agent_and_a_success_resets_them(ctx):
    guard = Guard(max_steps=50, max_seconds=60, max_errors_in_row=3)
    for _ in range(2):
        dispatch(ctx, ToolCall(id="x", name="analyze_document", args={}), guard)
    dispatch(ctx, call("list_inputs"), guard)
    assert not guard.stuck
    for _ in range(3):
        dispatch(ctx, ToolCall(id="x", name="analyze_document", args={}), guard)
    assert guard.stuck


def _scanned(ctx, file_name, text, conf):
    """Replace a document's page 1 with an OCR page of the given text and confidence."""
    from contract_analyst.agent.steps import resolve_doc_id
    from contract_analyst.models import PageMethod, PageQuality

    doc_id = resolve_doc_id(ctx, file_name)
    doc = ctx.documents[doc_id]
    page = doc.pages[0].model_copy(update={"method": PageMethod.OCR, "text": text, "quality": PageQuality(
        chars=len(text), ocr_mean_conf=conf, low_conf_ratio=0.0 if conf > 0.9 else 0.6)})
    return doc_id, doc.model_copy(update={"pages": [page], "status": "ok" if conf > 0.9 else "low_quality"})


def test_rerecognize_leaves_text_layer_pages_alone(ctx, monkeypatch):
    from contract_analyst.agent import steps

    ingest_all(ctx)
    monkeypatch.setattr(steps, "rerecognize_pages", lambda *a, **k: pytest.fail("a text page was sent to OCR"))
    result = dispatch(ctx, call("rerecognize", doc_id="Договор аренды.pdf", pages=[1], strategy="higher_dpi"),
                      Guard(10, 60))
    assert result.error_type == "not_ocr" and "текстового слоя" in result.hint


def test_rerecognize_reparses_the_header_of_the_better_text(ctx, monkeypatch):
    from contract_analyst.agent import steps

    ingest_all(ctx)
    doc_id, worse = _scanned(ctx, "ДС2 скан.pdf", "Доп0лн1тельн0е с0глашен1е", 0.4)
    ctx.documents[doc_id] = worse
    ctx.headers[doc_id] = steps.parse_header(worse.pages[0].text)
    _, better = _scanned(ctx, "ДС2 скан.pdf", "Дополнительное соглашение № D990000003-01\nк Договору аренды "
                                             "№ D990000001-01 от 10.01.2024 г.", 0.97)
    monkeypatch.setattr(steps, "rerecognize_pages", lambda *a, **k: better)
    result = dispatch(ctx, call("rerecognize", doc_id=doc_id, pages=[1], strategy="higher_dpi"), Guard(10, 60))
    assert result.data["adopted"] and ctx.headers[doc_id].number == "D990000003-01"


def test_header_reanalysis_updates_the_end_date(ctx, monkeypatch):
    from datetime import date

    import contract_analyst.agent.steps as steps_module
    from tests.fixtures.builders import make_record

    ends = iter([None, date(2026, 12, 31)])
    monkeypatch.setattr(steps_module, "understand_document", lambda doc, header, client, settings, groups=None:
                        make_record(doc.doc_id, "D990000002-01", effective_to=next(ends)))
    ingest_all(ctx)
    steps_module.analyze_docs(ctx, ["ДС1.pdf"])
    steps_module.analyze_docs(ctx, ["ДС1.pdf"], ["header"])
    (record,) = ctx.records.values()
    assert record.effective_to.value == date(2026, 12, 31)


def test_scan_read_returns_low_quality_with_hint(tmp_path, fixtures_package):
    ctx = make_ctx(tmp_path, fixtures_package, ocr=FakeOcrEngine(conf=0.3))
    result = dispatch(ctx, call("read_document", file_name="ДС2 скан.pdf"), Guard(10, 60))
    assert result.status == "low_quality" and "rerecognize" in result.hint


def test_tool_data_is_compact(tmp_path):
    make_text_pdf(tmp_path / "long.pdf", [f"{i}. Пункт договора номер {i} " + "текст " * 120 for i in range(1, 80)])
    ctx = make_ctx(tmp_path, tmp_path)
    result = dispatch(ctx, call("read_document", file_name="long.pdf"), Guard(10, 60))
    assert len(json.dumps(result.data, ensure_ascii=False)) <= 2000


def test_ingest_lists_duplicates_and_unsupported(tmp_path, fixtures_package):
    import shutil

    package = tmp_path / "pkg"
    shutil.copytree(fixtures_package, package)
    shutil.copy(package / "ДС1.pdf", package / "ДС1 (копия).pdf")
    (package / "t.xlsx").write_bytes(b"x")
    ctx = make_ctx(tmp_path, package)
    ingest_all(ctx)
    reasons = {u["file"]: u["reason"] for u in ctx.unprocessed}
    duplicates = [f for f in ("ДС1.pdf", "ДС1 (копия).pdf") if "дубликат" in reasons.get(f, "")]
    assert len(duplicates) == 1 and reasons["t.xlsx"] == "unsupported_format"
    assert len(ctx.documents) == 3 and set(ctx.headers) == set(ctx.documents)


def test_analyze_resolves_document_number_and_file_stem(ctx, monkeypatch):
    import contract_analyst.agent.steps as steps_module
    from tests.fixtures.builders import make_record

    monkeypatch.setattr(steps_module, "understand_document",
                        lambda doc, header, client, settings, groups=None: make_record(doc.doc_id, header.number, kind="main"))
    ingest_all(ctx)
    assert dispatch(ctx, call("analyze_document", doc_id="D990000002-01"), Guard(10, 60)).data["analyzed"]
    assert dispatch(ctx, call("analyze_document", doc_id="Договор аренды"), Guard(10, 60)).data["analyzed"]


def test_missing_thought_falls_back_to_message_text_and_is_flagged(ctx):
    guard = Guard(10, 60)
    result = dispatch(ctx, ToolCall(id="1", name="list_inputs", args={}), guard, message_text="Мысль: смотрю опись")
    assert result.status == "ok"
    no_text = dispatch(ctx, ToolCall(id="2", name="build_chains", args={}), guard)
    assert no_text.status in ("ok", "empty")
    thoughts = [e for e in read_trace(ctx.trace.path) if e["event"] == "thought"]
    assert thoughts[0]["thought"] == "смотрю опись" and thoughts[0]["thought_source"] == "message"
    assert thoughts[1]["thought_source"] == "missing"


def test_missing_thought_falls_back_to_model_reasoning(ctx):
    guard = Guard(10, 60)
    dispatch(ctx, ToolCall(id="1", name="list_inputs", args={}), guard, reasoning="Сначала смотрю опись файлов.")
    dispatch(ctx, ToolCall(id="2", name="build_chains", args={}), guard, reasoning="а" * 1000 + " Собираю цепочки.")
    thoughts = [e for e in read_trace(ctx.trace.path) if e["event"] == "thought"]
    assert thoughts[0]["thought"] == "Сначала смотрю опись файлов." and thoughts[0]["thought_source"] == "reasoning"
    assert thoughts[1]["thought"].endswith("Собираю цепочки.") and len(thoughts[1]["thought"]) <= 601
