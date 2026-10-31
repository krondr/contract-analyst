"""Held-out synthetic chains replayed from the recorded LLM cache: generalization check, no tuning here."""

from datetime import date
from pathlib import Path

import pytest

from contract_analyst.agent.runner import run_analysis
from contract_analyst.config import load_settings
from contract_analyst.eval.golden import load_golden
from contract_analyst.eval.metrics import score_run

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "synthetic/llm_cache.sqlite"


@pytest.mark.skipif(not CACHE.exists(), reason="LLM responses for synthetic data not recorded yet")
def test_heldout_synthetic_replay(tmp_path):
    settings = load_settings(llm_cache_mode="replay")
    result = run_analysis(ROOT / "synthetic/build", settings, date(2026, 10, 30), out_dir=tmp_path,
                          llm_cache=CACHE, read_cache=None, console=False)
    scores = score_run(result.run_dir, load_golden(ROOT / "synthetic/golden.yaml"))
    assert scores.critical_accepted_wrong == 0 and scores.chain_acc == 1.0, scores
