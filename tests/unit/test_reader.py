import cv2
import pymupdf

from contract_analyst.config import load_settings
from contract_analyst.ingest.inventory import list_inputs
from contract_analyst.ingest.reader import read_document, rerecognize_pages
from tests.fixtures.builders import (
    FakeOcrEngine,
    build_mixed_pdf,
    make_docx,
    make_scan_pdf,
    render_text_image,
)


def _read(path, engine=None, **kw):
    (f,) = [f for f in list_inputs(path.parent) if f.name == path.name]
    return read_document(f, engine or FakeOcrEngine(), load_settings(), **kw)


def test_mixed_pdf_routes_pages(tmp_path):
    doc = _read(build_mixed_pdf(tmp_path / "m.pdf"))
    assert [p.method for p in doc.pages] == ["text", "ocr", "empty"]
    assert doc.status == "ok" and doc.doc_id == doc.file.sha256[:12]
    assert "1500" in doc.pages[1].text and doc.pages[1].quality.ocr_mean_conf == 0.95


def test_low_ocr_confidence_marks_low_quality(tmp_path):
    doc = _read(make_scan_pdf(tmp_path / "s.pdf", ["Ставка 1500 рублей"]), FakeOcrEngine(conf=0.4))
    assert doc.status == "low_quality" and doc.pages[0].quality.low_conf_ratio == 1.0


def test_docx_and_image(tmp_path):
    make_docx(tmp_path / "d.docx", ["Договор № D990000001-01", "1. Предмет договора"])
    cv2.imwrite(str(tmp_path / "i.png"), render_text_image("Ставка 1500 рублей"))
    assert _read(tmp_path / "d.docx").status == "ok"
    assert [c.clause_id for c in _read(tmp_path / "d.docx").clauses] == ["preamble", "1"]
    assert _read(tmp_path / "i.png").pages[0].method == "ocr"


def test_encrypted_corrupted_unsupported(tmp_path):
    enc = pymupdf.open()
    enc.new_page()
    enc.save(tmp_path / "e.pdf", encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="u", owner_pw="o")
    (tmp_path / "c.pdf").write_bytes(b"%PDF-broken")
    (tmp_path / "t.xlsx").write_bytes(b"x")
    for name, issue in [("e.pdf", "encrypted"), ("c.pdf", "corrupted"), ("t.xlsx", "unsupported_format")]:
        doc = _read(tmp_path / name)
        assert doc.status == "error" and issue in doc.issues and doc.pages == []


def test_all_blank_pdf_is_empty(tmp_path):
    blank = pymupdf.open()
    blank.new_page()
    blank.save(tmp_path / "b.pdf")
    assert _read(tmp_path / "b.pdf").status == "empty"


def test_cache_reuses_result_without_ocr(tmp_path):
    build_mixed_pdf(tmp_path / "m.pdf")
    first = _read(tmp_path / "m.pdf", FakeOcrEngine(text="первый 1500"), cache_dir=tmp_path / "cache")
    again = _read(tmp_path / "m.pdf", FakeOcrEngine(text="второй"), cache_dir=tmp_path / "cache")
    assert again.pages[1].text == first.pages[1].text == "первый 1500"


def test_rerecognize_replaces_only_requested_pages(tmp_path):
    doc = _read(build_mixed_pdf(tmp_path / "m.pdf"), FakeOcrEngine(conf=0.4))
    better = rerecognize_pages(doc, [2], FakeOcrEngine(text="лучше 1500", conf=0.97), "none", 400)
    assert better.pages[1].text == "лучше 1500" and better.pages[0].text == doc.pages[0].text
    assert better.status == "ok"
