import json

from contract_analyst.agent.guards import Guard
from contract_analyst.agent.steps import ingest_all
from contract_analyst.agent.tools import TOOLS, dispatch, tool_schemas
from contract_analyst.agent.trace import read_trace
from contract_analyst.llm.client import ToolCall
from tests.agent.conftest import make_ctx
from tests.fixtures.builders import FakeOcrEngine, make_text_pdf


def call(name, **args):
    return ToolCall(id="c1", name=name, args={"thought": "проверяю шаг", **args})


def test_every_tool_requires_thought_and_has_schema():
    for spec in TOOLS.values():
        assert spec.args_model.model_fields["thought"].is_required()
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
    assert dispatch(ctx, ToolCall(id="x", name="read_document", args={"file_name": "a"}), Guard(10, 60)).error_type == "bad_args"
    assert dispatch(ctx, call("format_disk"), Guard(10, 60)).error_type == "unknown_tool"


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
