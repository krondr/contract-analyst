from pathlib import Path

from contract_analyst.eval.experiments import (
    VariantResult,
    VariantSpec,
    decide,
    select_e2_winner,
    select_e3_winner,
    select_e4_winner,
)
from contract_analyst.eval.metrics import Scores


def result(name, variant="B", **scores):
    return VariantResult(spec=VariantSpec(name=name, variant=variant), scores=Scores(**scores), run_dir=Path("x"))


def test_e2_errors_disqualify_then_coverage_decides():
    rows = [result("ollama:qwen3:14b", critical_accepted_wrong=1, critical_coverage=0.95),
            result("ollama:qwen3:8b", critical_accepted_wrong=0, critical_coverage=0.80)]
    assert select_e2_winner(rows).spec.name == "ollama:qwen3:8b"


def test_e2_coverage_tie_goes_to_validity_then_speed_then_smaller_model():
    rows = [result("ollama:qwen3:14b", critical_coverage=0.90, json_valid_rate=1.0, tool_call_valid_rate=1.0, wall_seconds=900),
            result("ollama:qwen3:8b", critical_coverage=0.89, json_valid_rate=0.9, tool_call_valid_rate=1.0, wall_seconds=500)]
    assert select_e2_winner(rows).spec.name == "ollama:qwen3:14b"
    same = [result("ollama:qwen3:14b", critical_coverage=0.9, wall_seconds=500),
            result("ollama:qwen3:8b", critical_coverage=0.9, wall_seconds=500)]
    assert select_e2_winner(same).spec.name == "ollama:qwen3:8b"


def test_e3_code_wins_unless_llm_clearly_better():
    assert select_e3_winner([result("code", consolidation_acc=0.85), result("llm", consolidation_acc=0.95)]).spec.name == "code"
    assert select_e3_winner([result("code", consolidation_acc=0.60), result("llm", consolidation_acc=0.95)]).spec.name == "llm"


def test_e4_only_agents_can_win_and_tie_goes_to_hybrid():
    rows = [result("A", "A", traps_passed=16, consolidation_acc=1.0), result("B", "B", traps_passed=14),
            result("C", "C", traps_passed=14), result("S", "S", traps_passed=16)]
    assert select_e4_winner(rows).spec.name == "B"
    rows.append(result("C2", "C", traps_passed=15))
    assert select_e4_winner(rows).spec.name == "C2"


def test_decide_returns_rule_text():
    _title, rule, winner = decide("e2", [result("ollama:qwen3:8b")])
    assert winner == "ollama:qwen3:8b" and "неверн" in rule


def test_each_golden_date_is_scored_from_its_own_run(tmp_path, monkeypatch):
    import json
    from datetime import date
    from types import SimpleNamespace

    from pydantic import TypeAdapter

    from contract_analyst.eval.experiments import run_variant
    from contract_analyst.models import ConsolidatedContract
    from tests.unit.test_metrics import AS_OF, MAIN, golden, make_run

    later = date(2027, 2, 1)
    gs = golden()
    gs.as_of_dates.append(later)
    gs.chains[0].status_at[later] = "partially_suspended"  # what the code fold computes at that date
    dates = []

    def llm_fold_run(data_dir, settings, as_of, variant, out_dir, console, **kw):
        dates.append(as_of)
        run_dir = make_run(out_dir / f"run{len(dates)}")
        wrong = {MAIN: ConsolidatedContract(contract_number=MAIN, as_of=as_of, status="terminated")}
        (run_dir / "kb" / "consolidated.json").write_bytes(TypeAdapter(dict[str, ConsolidatedContract]).dump_json(wrong))
        (run_dir / "metrics.json").write_text(json.dumps({"as_of": as_of.isoformat()}))
        return SimpleNamespace(run_dir=run_dir)

    monkeypatch.setattr("contract_analyst.agent.runner.run_analysis", llm_fold_run)
    result = run_variant(VariantSpec(name="llm", overrides={"consolidation_mode": "llm"}), tmp_path, gs, AS_OF,
                         out_dir=tmp_path)
    assert dates == [AS_OF, later] and set(result.runs_at) == {AS_OF, later}
    assert result.scores.consolidation_acc == 0.0
