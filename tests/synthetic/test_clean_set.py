"""Clean evaluation sets: the build is reproducible, golden answers agree with the built files, all of it fictional.

Only the golden sets themselves are checked here. The code fold is not compared with them (fold_mismatches) and the
agent is not run on them: each set measures a result once (clean/README.md, clean2/README.md,
clean3/README.md)."""

import hashlib
import importlib
import shutil
from pathlib import Path

import pytest

from contract_analyst.config import load_settings
from contract_analyst.eval.golden import load_golden
from contract_analyst.eval.golden_tools import check_golden
from contract_analyst.ingest.inventory import list_inputs
from contract_analyst.ingest.reader import read_document
from contract_analyst.ocr.tesseract import TesseractEngine
from contract_analyst.parsing.patterns import DOC_NUMBER_RE

ROOT = Path(__file__).resolve().parents[2]
needs_tesseract = pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract binary missing")

# (set directory, build script module, prefix of every document number in the set)
SETS = [("clean", "make_clean", "D97"), ("clean2", "make_clean2", "D96"), ("clean3", "make_clean3", "D95")]
by_set = pytest.mark.parametrize(("name", "script", "prefix"), SETS, ids=[s[0] for s in SETS])


@pytest.mark.ocr
@needs_tesseract
@by_set
def test_clean_golden_consistent_and_files_readable(name, script, prefix):
    root = ROOT / name
    docs = {d.doc_id: d for d in (read_document(f, TesseractEngine(), load_settings())
                                  for f in list_inputs(root / "build"))}
    assert check_golden(load_golden(root / "golden.yaml"), docs) == []
    assert all(d.status in ("ok", "low_quality") for d in docs.values())


@by_set
def test_clean_golden_covers_every_built_file_on_both_dates(name, script, prefix):
    root = ROOT / name
    gs = load_golden(root / "golden.yaml")
    built = {f.sha256[:12] for f in list_inputs(root / "build")}
    assert {d.doc_id for d in gs.documents} == built == {doc_id for c in gs.chains for doc_id in c.order}
    assert {d.file for d in gs.documents} == set(_digests(root / "build"))
    for chain in gs.chains:
        assert set(chain.status_at) == set(chain.terms_at) == set(chain.rates_at) == set(gs.as_of_dates)


@by_set
def test_clean_only_fictional_document_numbers(name, script, prefix):
    root = ROOT / name
    sources = [p for p in root.rglob("*") if p.suffix in (".md", ".yaml")]
    numbers = {n for p in sources for n in DOC_NUMBER_RE.findall(p.read_text(encoding="utf-8"))}
    numbers |= {n for p in (root / "build").rglob("*") for n in DOC_NUMBER_RE.findall(p.name)}
    assert numbers and all(n.startswith(prefix) for n in numbers)


def _digests(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file() and not p.name.startswith(".")}


@by_set
def test_clean_build_is_byte_reproducible(name, script, prefix, tmp_path, monkeypatch):
    """Two builds give the same bytes. Text PDFs and DOCX also match the committed files; scan pages go through the
    platform's JPEG encoder, so a scan is compared only between two builds on one machine."""
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    make = importlib.import_module(script)

    builds = []
    for run in ("a", "b"):
        monkeypatch.setattr(make, "BUILD", tmp_path / run)
        monkeypatch.setattr(make, "GOLDEN", tmp_path / "absent.yaml")
        make.main()
        builds.append(_digests(tmp_path / run))
    assert builds[0] == builds[1]
    from make_synthetic import Item

    scans = {item.output for item in make.manifest(Item) if item.kind == "scan_pdf"}
    committed = _digests(ROOT / name / "build")
    assert set(builds[0]) == set(committed)
    assert {k: v for k, v in builds[0].items() if k not in scans} == {k: v for k, v in committed.items()
                                                                       if k not in scans}
