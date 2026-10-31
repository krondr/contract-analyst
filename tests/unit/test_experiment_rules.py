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
