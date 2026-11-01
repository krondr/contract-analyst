"""The Reflection table: the planner's own items with the reflector's verdict, and checks computed, not asserted."""

import json
from datetime import date
from pathlib import Path

import pytest

from contract_analyst.agent import steps
from contract_analyst.agent.runner import run_analysis
from contract_analyst.config import load_settings
from contract_analyst.llm.scripted import ScriptedClient, load_script
from tests.agent.test_hybrid import RECORDS
from tests.fixtures.builders import FakeOcrEngine
from tests.fixtures.builders import make_record as rec

HAPPY = Path(__file__).parent / "scripts" / "happy_path.json"
KB_ITEM = "Непроверенные значения не записаны в базу знаний"


@pytest.fixture(autouse=True)
def stub_understanding(monkeypatch):
    def fake(doc, header, client, settings, groups=None):
        spec = dict(RECORDS[doc.file.name])
        return rec(doc.doc_id, spec.pop("number"), file_name=doc.file.name, **spec)

    monkeypatch.setattr(steps, "understand_document", fake)


def run(package, tmp_path, script):
    r = run_analysis(package, load_settings(), date(2026, 10, 30), client=ScriptedClient(script),
                     ocr_engine=FakeOcrEngine(), out_dir=tmp_path, read_cache=None, console=False)
    return r.report_path.read_text(encoding="utf-8").split("## 7. Reflection")[1].split("## 8.")[0]


def row(reflection: str, item: str) -> str:
    return next(line for line in reflection.splitlines() if line.startswith(f"| {item} |"))


def test_unverified_values_check_reads_the_written_knowledge_base(tmp_path, fixtures_package, monkeypatch):
    assert "✅" in row(run(fixtures_package, tmp_path / "a", load_script(HAPPY)), KB_ITEM)
    from contract_analyst.kb import store

    monkeypatch.setattr(store, "_scrub_record", lambda r: r)  # a store that keeps a rejected value
    leaked = row(run(fixtures_package, tmp_path / "b", load_script(HAPPY)), KB_ITEM)
    assert "❌" in leaked and "doc_date" in leaked


def test_planner_items_are_shown_with_the_reflector_verdict(tmp_path, fixtures_package):
    script = load_script(HAPPY)
    script[0] = json.dumps({"items": ["Перераспознать скан", "Сверить даты"]}, ensure_ascii=False)
    script[-1] = json.dumps({"goal_met": False, "gaps": [], "items": [
        {"n": 1, "done": True, "note": "скан перераспознан"}, {"n": 2, "done": False, "note": "даты не сверены"}]},
        ensure_ascii=False)
    reflection = run(fixtures_package, tmp_path, script)
    assert "✅" in row(reflection, "Перераспознать скан") and "скан перераспознан" in row(reflection,
                                                                                          "Перераспознать скан")
    assert "❌" in row(reflection, "Сверить даты") and "даты не сверены" in row(reflection, "Сверить даты")


def test_report_must_have_all_eight_sections(tmp_path, fixtures_package, monkeypatch):
    render = steps.render_report
    monkeypatch.setattr(steps, "render_report", lambda ctx: render(ctx).split("## 8.")[0])
    reflection = run_analysis(fixtures_package, load_settings(), date(2026, 10, 30),
                              client=ScriptedClient(load_script(HAPPY)), ocr_engine=FakeOcrEngine(), out_dir=tmp_path,
                              read_cache=None, console=False).report_path.read_text(encoding="utf-8")
    assert "❌" in row(reflection.split("## 7. Reflection")[1], "Все разделы отчёта на месте")
