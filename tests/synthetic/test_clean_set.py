"""Clean evaluation set: the build is reproducible, golden answers agree with the built files, all of it fictional.

Only the golden set itself is checked here. The code fold is not compared with it (fold_mismatches) and the agent is
not run on it: the set measures the result of a round of rule changes once, after the round (clean/README.md)."""

import hashlib
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
CLEAN = ROOT / "clean"
needs_tesseract = pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract binary missing")


@pytest.mark.ocr
@needs_tesseract
def test_clean_golden_consistent_and_files_readable():
    docs = {d.doc_id: d for d in (read_document(f, TesseractEngine(), load_settings())
                                  for f in list_inputs(CLEAN / "build"))}
    assert check_golden(load_golden(CLEAN / "golden.yaml"), docs) == []
    assert all(d.status in ("ok", "low_quality") for d in docs.values())


def test_clean_golden_covers_every_built_file_on_both_dates():
    gs = load_golden(CLEAN / "golden.yaml")
    built = {f.sha256[:12] for f in list_inputs(CLEAN / "build")}
    assert {d.doc_id for d in gs.documents} == built == {doc_id for c in gs.chains for doc_id in c.order}
    assert {d.file for d in gs.documents} == set(_digests(CLEAN / "build"))
    for chain in gs.chains:
        assert set(chain.status_at) == set(chain.terms_at) == set(chain.rates_at) == set(gs.as_of_dates)


def test_clean_only_fictional_document_numbers():
    sources = [p for p in CLEAN.rglob("*") if p.suffix in (".md", ".yaml")]
    numbers = {n for p in sources for n in DOC_NUMBER_RE.findall(p.read_text(encoding="utf-8"))}
    numbers |= {n for p in (CLEAN / "build").rglob("*") for n in DOC_NUMBER_RE.findall(p.name)}
    assert numbers and all(n.startswith("D97") for n in numbers)


def _digests(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file() and not p.name.startswith(".")}


def test_clean_build_is_byte_reproducible(tmp_path, monkeypatch):
    """Two builds give the same bytes. Text PDFs and DOCX also match the committed files; scan pages go through the
    platform's JPEG encoder, so a scan is compared only between two builds on one machine."""
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import make_clean

    builds = []
    for name in ("a", "b"):
        monkeypatch.setattr(make_clean, "BUILD", tmp_path / name)
        monkeypatch.setattr(make_clean, "GOLDEN", tmp_path / "absent.yaml")
        make_clean.main()
        builds.append(_digests(tmp_path / name))
    assert builds[0] == builds[1]
    from make_synthetic import Item

    scans = {item.output for item in make_clean.manifest(Item) if item.kind == "scan_pdf"}
    committed = _digests(CLEAN / "build")
    assert set(builds[0]) == set(committed)
    assert {k: v for k, v in builds[0].items() if k not in scans} == {k: v for k, v in committed.items()
                                                                       if k not in scans}
