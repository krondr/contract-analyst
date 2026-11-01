import json
from datetime import date
from pathlib import Path

import pytest

from contract_analyst.agent import steps
from contract_analyst.agent.runner import run_analysis
from contract_analyst.agent.trace import read_trace
from contract_analyst.config import load_settings
from contract_analyst.llm.client import AssistantTurn, ToolCall
from contract_analyst.llm.scripted import ScriptedClient, load_script
from tests.fixtures.builders import FakeOcrEngine, make_text_pdf
from tests.fixtures.builders import make_record as rec

SCRIPTS = Path(__file__).parent / "scripts"
MAIN = "D990000001-01"
RECORDS = {
    "Договор аренды.pdf": {"number": MAIN, "kind": "main", "doc_date": date(2024, 1, 10), "category": "lease",
                               "rates": [("Аренда", 100000, None, None)]},
    "ДС1.pdf": {"number": "D990000002-01", "parent": MAIN, "effective": date(2025, 4, 1), "rates": [("Аренда", 120000, None, None)]},
    "ДС2 скан.pdf": {"number": "D990000003-01", "parent": MAIN, "rejected_date": date(2025, 10, 1),
                         "terms": {"validity": "до 31.12.2026"}},
    "Другой.pdf": {"number": "D990000009-01", "kind": "main", "doc_date": date(2023, 1, 1), "category": "services"},
}


@pytest.fixture(autouse=True)
def stub_understanding(monkeypatch):
    def fake(doc, header, client, settings, groups=None):
        spec = dict(RECORDS[doc.file.name])
        return rec(doc.doc_id, spec.pop("number"), file_name=doc.file.name, **spec)

    monkeypatch.setattr(steps, "understand_document", fake)


def run(package, script, tmp_path, **kw):
    return run_analysis(package, kw.pop("settings", load_settings()), date(2026, 10, 30),
                        client=kw.pop("client", None) or ScriptedClient(script),
                        ocr_engine=kw.pop("ocr", FakeOcrEngine()), out_dir=tmp_path, read_cache=None, **kw)


def chat(*calls, content=""):
    return AssistantTurn(content=content, tool_calls=[ToolCall(id=str(i), name=n, args={"thought": "шаг", **a})
                                                      for i, (n, a) in enumerate(calls)])


def test_hybrid_produces_report_and_thought_for_every_call(tmp_path, fixtures_package):
    r = run(fixtures_package, load_script(SCRIPTS / "happy_path.json"), tmp_path)
    report = r.report_path.read_text(encoding="utf-8")
    assert "## 7. Reflection" in report and "120 000" in report
    trace = read_trace(r.trace_path)
    calls = [e for e in trace if e["event"] == "tool_call"]
    assert len(calls) == 7 and all(e["thought"] for e in calls)
    assert [e["event"] for e in trace].count("plan") == 1
    metrics = json.loads((r.run_dir / "metrics.json").read_text())
    assert metrics["variant"] == "B" and metrics["safety_net"] == [] and metrics["thought_coverage"] == 1.0
    assert metrics["scope"] == {"chain": None, "doc": None}
    assert "| Мысль записана у каждого вызова инструмента | ✅ | 7 из 7 |" in report


def test_recorded_run_replays_without_misses(tmp_path, fixtures_package):
    from contract_analyst.llm.cache import CachingClient

    db = tmp_path / "cache.sqlite"
    recorder = CachingClient(ScriptedClient(load_script(SCRIPTS / "happy_path.json")), db, "record")
    first = run_analysis(fixtures_package, load_settings(), date(2026, 10, 30), client=recorder,
                         ocr_engine=FakeOcrEngine(), out_dir=tmp_path / "runs", read_cache=None, console=False)
    replay = CachingClient(ScriptedClient([]), db, "replay")
    second = run_analysis(fixtures_package, load_settings(), date(2026, 10, 30), client=replay,
                          ocr_engine=FakeOcrEngine(), out_dir=tmp_path / "runs", read_cache=None, console=False)
    assert second.run_dir != first.run_dir and replay.hits == recorder.inner.usage["calls"] == 9


class Looping:
    """A model stuck in a loop: every turn it repeats the same call; the reflector always sees gaps."""

    model_name = "looping"

    def __init__(self, turn):
        self.turn, self.usage = turn, {"calls": 0}

    def chat(self, messages, tools=None, think=False):
        self.usage["calls"] += 1
        return self.turn

    def structured(self, messages, schema):
        self.usage["calls"] += 1
        return {"PlanOut": '{"items": ["План"]}', "ReflectionOut": '{"goal_met": false, "gaps": ["не всё"]}'}.get(
            schema.__name__, '{"summary": "Аренда."}')


def test_model_looping_on_invalid_calls_is_stopped_and_the_report_written(tmp_path, fixtures_package):
    r = run(fixtures_package, [], tmp_path, client=Looping(chat(("analyze_document", {}))))
    errors = [e for e in read_trace(r.trace_path) if e["event"] == "tool_result" and e["status"] == "error"]
    assert len(errors) == 5 * 2  # the error cap per executor round, two rounds of reflection
    assert "### D990000001-01" in r.report_path.read_text(encoding="utf-8")


def test_recursion_limit_still_ends_in_a_report(tmp_path, fixtures_package):
    r = run(fixtures_package, [], tmp_path, client=Looping(AssistantTurn(content="Думаю")),
            settings=load_settings(max_reflection_rounds=1000, max_steps=10))
    assert any("GraphRecursionError" in str(e.get("message")) for e in read_trace(r.trace_path))
    assert "### D990000001-01" in r.report_path.read_text(encoding="utf-8")


def test_low_quality_scan_triggers_rerecognize_and_refusal_is_reported(tmp_path, fixtures_package):
    script = [{"items": ["Перераспознать скан"]}, chat(("rerecognize", {"doc_id": "ДС2 скан.pdf", "pages": [1],
                                                                         "strategy": "higher_dpi"})),
              chat(content="Готово"), '{"goal_met": true, "gaps": []}', chat(content="Готово"),
              '{"goal_met": true, "gaps": []}', '{"summary": "Договор аренды."}']
    script[0] = json.dumps(script[0], ensure_ascii=False)
    r = run(fixtures_package, script, tmp_path, ocr=FakeOcrEngine(conf=0.3))
    assert any(e["event"] == "tool_call" and e["tool"] == "rerecognize" for e in read_trace(r.trace_path))
    refusals = r.report_path.read_text(encoding="utf-8").split("## 5. Отказы и противоречия")[1].split("## 6.")[0]
    assert "doc_date" in refusals


def test_reflection_loops_at_most_twice_then_safety_net_completes(tmp_path, fixtures_package):
    gaps = '{"goal_met": false, "gaps": ["документы не разобраны"]}'
    script = ['{"items": ["План"]}', chat(content="Готово"), gaps, chat(content="Готово"), gaps,
              '{"summary": "Договор аренды."}']
    r = run(fixtures_package, script, tmp_path)
    assert [e["event"] for e in read_trace(r.trace_path)].count("reflection") == 2
    metrics = json.loads((r.run_dir / "metrics.json").read_text())
    assert "analyze_docs" in metrics["safety_net"]
    assert "❌" in r.report_path.read_text(encoding="utf-8").split("## 7. Reflection")[1]


def test_only_chain_limits_analysis(tmp_path, fixtures_package):
    import shutil

    package = tmp_path / "pkg"
    shutil.copytree(fixtures_package, package)
    make_text_pdf(package / "Другой.pdf", ["Договор № D990000009-01\nг. Псков 01.01.2023 г.\n1. Предмет договора."])
    script = ['{"items": ["План"]}', chat(("analyze_document", {"doc_id": "Другой.pdf"})), chat(content="Готово"),
              '{"goal_met": true, "gaps": []}', chat(content="Готово"), '{"goal_met": true, "gaps": []}',
              '{"summary": "Аренда."}']
    r = run(package, script, tmp_path, only_chain=MAIN)
    assert json.loads((r.run_dir / "metrics.json").read_text())["scope"] == {"chain": MAIN, "doc": None}
    result = next(e for e in read_trace(r.trace_path) if e["event"] == "tool_result" and e["tool"] == "analyze_document")
    assert "skipped" in result["data"] and "Другой" not in r.report_path.read_text(encoding="utf-8").split("## 4.")[1]


def test_executor_thinks_and_its_reasoning_becomes_the_thought(tmp_path, fixtures_package):
    turn = AssistantTurn(reasoning="Сначала соберу цепочки.", tool_calls=[ToolCall(id="0", name="build_chains", args={})])
    reflect = '{"goal_met": true, "gaps": []}'
    client = ScriptedClient(['{"items": ["План"]}', turn, chat(content="Готово"), reflect, chat(content="Готово"),
                             reflect, '{"summary": "Аренда."}'])
    r = run_analysis(fixtures_package, load_settings(agent_thinking=True), date(2026, 10, 30), client=client,
                     ocr_engine=FakeOcrEngine(), out_dir=tmp_path, read_cache=None)
    assert all(c["think"] for c in client.calls if c["kind"] == "chat")
    trace = read_trace(r.trace_path)
    assert any(e["event"] == "llm_call" and e.get("reasoning") == "Сначала соберу цепочки." for e in trace)
    call = next(e for e in trace if e["event"] == "tool_call")
    assert call["thought"] == "Сначала соберу цепочки." and call["thought_source"] == "reasoning"


def test_calls_without_any_thought_lower_the_coverage(tmp_path, fixtures_package):
    silent = AssistantTurn(tool_calls=[ToolCall(id="0", name="build_chains", args={})])
    reflect = '{"goal_met": true, "gaps": []}'
    script = ['{"items": ["План"]}', silent, chat(("consolidate", {})), chat(content="Готово"), reflect,
              chat(content="Готово"), reflect, '{"summary": "Аренда."}']
    r = run(fixtures_package, script, tmp_path, settings=load_settings(agent_thinking=False))
    assert json.loads((r.run_dir / "metrics.json").read_text())["thought_coverage"] == 0.5
    assert "| Мысль записана у каждого вызова инструмента | ❌ | 1 из 2 |" in r.report_path.read_text(encoding="utf-8")
