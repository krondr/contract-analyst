"""Held-out synthetic set: golden answers agree with the built files; the files are readable and fictional."""

import shutil
from pathlib import Path

import pytest
import yaml

from contract_analyst.config import load_settings
from contract_analyst.eval.golden import load_golden
from contract_analyst.eval.golden_tools import check_golden, fold_mismatches
from contract_analyst.ingest.inventory import list_inputs
from contract_analyst.ingest.reader import read_document
from contract_analyst.ocr.tesseract import TesseractEngine
from contract_analyst.parsing.patterns import DOC_NUMBER_RE

ROOT = Path(__file__).resolve().parents[2]
needs_tesseract = pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract binary missing")


@pytest.mark.ocr
@needs_tesseract
def test_synthetic_golden_consistent_and_files_readable():
    gs = load_golden(ROOT / "synthetic/golden.yaml")
    docs = {d.doc_id: d for d in (read_document(f, TesseractEngine(), load_settings())
                                  for f in list_inputs(ROOT / "synthetic/build"))}
    assert check_golden(gs, docs) == []
    assert all(d.status in ("ok", "low_quality") for d in docs.values())


@pytest.mark.xfail(strict=True, reason="needs adjudication: ДС3 has no replacing op, so the code fold joins the old and "
                                      "the new plan_indicators; the golden keeps only «не применяется с 01.03.2026»")
def test_code_fold_agrees_with_golden():
    assert fold_mismatches(load_golden(ROOT / "synthetic/golden.yaml")) == []


def test_golden_covers_every_built_file_on_both_dates():
    gs = load_golden(ROOT / "synthetic/golden.yaml")
    built = {f.sha256[:12] for f in list_inputs(ROOT / "synthetic/build")}
    assert {d.doc_id for d in gs.documents} == built == {doc_id for c in gs.chains for doc_id in c.order}
    for chain in gs.chains:
        assert set(chain.status_at) == set(chain.terms_at) == set(chain.rates_at) == set(gs.as_of_dates)


def test_only_fictional_document_numbers():
    sources = [p for p in (ROOT / "synthetic").rglob("*") if p.suffix in (".md", ".yaml")]
    numbers = {n for p in sources for n in DOC_NUMBER_RE.findall(p.read_text(encoding="utf-8"))}
    numbers |= {n for p in (ROOT / "synthetic/build").rglob("*") for n in DOC_NUMBER_RE.findall(p.name)}
    assert numbers and all(n.startswith("D99") for n in numbers)


def test_chat_questions_well_formed():
    questions = yaml.safe_load((ROOT / "synthetic/chat_questions.yaml").read_text(encoding="utf-8"))
    ids = [q["id"] for q in questions]
    assert len(questions) == 8 and len(set(ids)) == 8
    for i, q in enumerate(questions):
        assert set(q) == {"id", "question", "expected_facts", "unanswerable", "follow_up_of"}
        assert q["follow_up_of"] in (None, *ids[:i])
        assert bool(q["expected_facts"]) != q["unanswerable"]
    assert any(q["unanswerable"] for q in questions) and any(q["follow_up_of"] for q in questions)
