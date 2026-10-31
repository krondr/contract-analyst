"""Checks against the real archive. Local only: data/ is never committed, so these skip elsewhere."""

import os
from pathlib import Path

import pytest

from contract_analyst.eval.golden import load_golden
from contract_analyst.eval.metrics import score_run
from contract_analyst.eval.traps import check_traps

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "data/golden/golden.yaml"
pytestmark = [pytest.mark.real, pytest.mark.skipif(not GOLDEN.exists(), reason="no golden set in data/golden")]


def final_run() -> Path:
    if run := os.environ.get("CA_FINAL_RUN"):
        return Path(run)
    runs = sorted(p for p in (ROOT / "runs").glob("*") if (p / "kb").exists())
    if not runs:
        pytest.skip("no finished run in runs/")
    return runs[-1]


def test_all_traps_pass_on_final_run():
    failed = [f"{t.trap_id}: {t.detail}" for t in check_traps(final_run(), load_golden(GOLDEN)) if not t.passed]
    assert not failed, failed


def test_golden_thresholds():
    s = score_run(final_run(), load_golden(GOLDEN))
    assert s.critical_accepted_wrong == 0, s
    assert s.critical_coverage >= 0.80 and s.consolidation_acc >= 0.90, s
