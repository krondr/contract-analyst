import json
from datetime import date

import pytest

from contract_analyst.agent import steps
from contract_analyst.agent.runner import run_analysis
from contract_analyst.agent.single_call import run_single_call
from contract_analyst.agent.trace import read_trace
from contract_analyst.config import load_settings
from contract_analyst.kb.store import KnowledgeBase
from contract_analyst.llm.client import AssistantTurn, ToolCall
from contract_analyst.llm.scripted import ScriptedClient
from tests.agent.test_hybrid import RECORDS
from tests.fixtures.builders import FakeOcrEngine
from tests.fixtures.builders import make_record as rec

AS_OF = date(2026, 10, 30)


@pytest.fixture(autouse=True)
def stub_understanding(monkeypatch):
    def fake(doc, header, client, settings, groups=None):
        spec = dict(RECORDS[doc.file.name])
        return rec(doc.doc_id, spec.pop("number"), file_name=doc.file.name, **spec)

    monkeypatch.setattr(steps, "understand_document", fake)


def turn(name, **args):
    return AssistantTurn(tool_calls=[ToolCall(id=name, name=name, args={"thought": f"шаг {name}", **args})])


def test_pipeline_variant_runs_without_planner_or_agent(tmp_path, fixtures_package):
    r = run_analysis(fixtures_package, load_settings(), AS_OF, "A", client=ScriptedClient(['{"summary": "Аренда."}']),
                     ocr_engine=FakeOcrEngine(), out_dir=tmp_path, read_cache=None, console=False)
    events = [e["event"] for e in read_trace(r.trace_path)]
    assert "plan" not in events and "tool_call" not in events and events.count("step") >= 5
    assert "### D990000001-01" in r.report_path.read_text(encoding="utf-8")


def test_full_agent_reads_files_itself(tmp_path, fixtures_package):
    names = ["Договор аренды.pdf", "ДС1.pdf", "ДС2 скан.pdf"]
    script = ['{"items": ["Прочитать и разобрать"]}', turn("list_inputs")]
    script += [turn("read_document", file_name=n) for n in names]
    script += [turn("analyze_document", doc_id=n) for n in names]
    script += [turn("build_chains"), turn("consolidate"), turn("cluster_and_compare"), '{"summary": "Аренда."}',
               turn("write_report"), AssistantTurn(content="Готово"), '{"goal_met": true, "gaps": []}']
    r = run_analysis(fixtures_package, load_settings(), AS_OF, "C", client=ScriptedClient(script),
                     ocr_engine=FakeOcrEngine(), out_dir=tmp_path, read_cache=None, console=False)
    reads = [e for e in read_trace(r.trace_path) if e["event"] == "tool_call" and e["tool"] == "read_document"]
    assert len(reads) == 3 and all(e["thought"] for e in reads)
    listed = next(e for e in read_trace(r.trace_path) if e["event"] == "tool_result" and e["tool"] == "list_inputs")
    assert '"documents"' not in listed["data"]
    assert json.loads((r.run_dir / "metrics.json").read_text())["safety_net"] == []


def test_single_call_baseline(tmp_path, fixtures_package):
    answer = {"documents": [{"file": "Договор аренды.pdf", "number": "D990000001-01", "kind": "main",
                             "doc_date": "2024-01-10"},
                            {"file": "ДС1.pdf", "number": "D990000002-01", "kind": "supplement",
                             "parent_number": "D990000001-01", "effective_from": "2025-04-01"}],
              "contracts": [{"number": "D990000001-01", "status": "active", "terms": {"validity": "до 31.12.2025"},
                             "rates": [{"label": "Аренда", "amount": "120000"}]}]}
    r = run_single_call(fixtures_package, load_settings(), AS_OF, tmp_path,
                        client=ScriptedClient([json.dumps(answer, ensure_ascii=False)]), ocr_engine=FakeOcrEngine(),
                        read_cache=None)
    kb = KnowledgeBase(r.kb_dir)
    contract = kb.get_contract_terms("D990000001-01")
    assert contract.status == "active" and [str(x.amount) for x in contract.rates] == ["120000"]
    assert r.variant == "S" and r.report_path.exists()
