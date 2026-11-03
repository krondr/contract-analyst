"""Synthetic chains replayed from the recorded LLM cache: a regression check without a model.

The set was used for error analysis after the first runs, so it is not strictly held out; the blind held-out set is
holdout/ (docs/experiments/final.md)."""

from datetime import date
from pathlib import Path

import pytest

from contract_analyst.agent.runner import run_analysis
from contract_analyst.config import load_settings
from contract_analyst.eval.golden import load_golden
from contract_analyst.eval.metrics import score_run

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "synthetic/llm_cache.sqlite"


@pytest.fixture(scope="module")
def scores(tmp_path_factory):
    if not CACHE.exists():
        pytest.skip("LLM responses for synthetic data not recorded yet")
    settings = load_settings(llm_cache_mode="replay")
    result = run_analysis(ROOT / "synthetic/build", settings, date(2026, 10, 30), out_dir=tmp_path_factory.mktemp("run"),
                          llm_cache=CACHE, read_cache=None, console=False)
    return score_run(result.run_dir, load_golden(ROOT / "synthetic/golden.yaml"))


def test_synthetic_replay_builds_the_right_structure(scores):
    assert scores.chain_acc == 1.0 and scores.doc_kind_acc == 1.0, scores


@pytest.mark.xfail(strict=True, reason="known gap (docs/experiments/final.md): 3 accepted wrong critical values "
                                      "with qwen3:8b; the requirement is zero")
def test_synthetic_replay_accepts_no_wrong_critical_value(scores):
    assert scores.critical_accepted_wrong == 0, scores
