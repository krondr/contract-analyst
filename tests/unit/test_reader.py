import cv2
import pymupdf

from contract_analyst.config import load_settings
from contract_analyst.ingest import reader
from contract_analyst.ingest.inventory import list_inputs
from contract_analyst.ingest.reader import read_document, rerecognize_pages
from contract_analyst.models import Clause
from tests.fixtures.builders import (
    FakeOcrEngine,
    build_mixed_pdf,
    make_docx,
    make_multipage_tiff,
    make_scan_pdf,
    render_text_image,
)


def _read(path, engine=None, settings=None, **kw):
    (f,) = [f for f in list_inputs(path.parent) if f.name == path.name]
    return read_document(f, engine or FakeOcrEngine(), settings or load_settings(), **kw)


class SizeRecordingOcr(FakeOcrEngine):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.heights: list[int] = []

    def recognize(self, image):
        self.heights.append(image.shape[0])
        return super().recognize(image)


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


def test_cache_key_covers_reader_version_and_the_settings_that_change_the_reading(tmp_path, monkeypatch):
    build_mixed_pdf(tmp_path / "m.pdf")
    cache = tmp_path / "cache"
    first = _read(tmp_path / "m.pdf", FakeOcrEngine(text="первый 1500"), cache_dir=cache)
    strict = _read(tmp_path / "m.pdf", FakeOcrEngine(text="второй 1500"), load_settings(ocr_word_conf_threshold=0.99),
                   cache_dir=cache)
    assert first.status == "ok" and strict.status == "low_quality" and strict.pages[1].text == "второй 1500"
    monkeypatch.setattr(reader, "READER_VERSION", reader.READER_VERSION + 1)
    assert _read(tmp_path / "m.pdf", FakeOcrEngine(text="третий 1500"), cache_dir=cache).pages[1].text == "третий 1500"


def test_cached_reading_is_segmented_again(tmp_path, monkeypatch):
    build_mixed_pdf(tmp_path / "m.pdf")
    _read(tmp_path / "m.pdf", cache_dir=tmp_path / "cache")
    monkeypatch.setattr(reader, "segment_clauses", lambda pages: [Clause(clause_id="новая", page=1, text="нарезка")])
    assert [c.clause_id for c in _read(tmp_path / "m.pdf", cache_dir=tmp_path / "cache").clauses] == ["новая"]


def test_multipage_tiff_reads_every_frame(tmp_path):
    texts = ["\n".join(["Страница договора"] * n) for n in (8, 11, 14)]  # frames of different heights
    make_multipage_tiff(tmp_path / "скан.tiff", texts)
    ocr = SizeRecordingOcr()
    doc = _read(tmp_path / "скан.tiff", ocr)
    assert [(p.number, p.method) for p in doc.pages] == [(1, "ocr"), (2, "ocr"), (3, "ocr")]
    assert len(set(ocr.heights)) == 3
    again = SizeRecordingOcr(text="лучше 1500")
    better = rerecognize_pages(doc, [2], again, "none", 300)
    assert again.heights == [ocr.heights[1]] and better.pages[1].text == "лучше 1500"


def test_rerecognize_replaces_only_requested_pages(tmp_path):
    doc = _read(build_mixed_pdf(tmp_path / "m.pdf"), FakeOcrEngine(conf=0.4))
    better = rerecognize_pages(doc, [2], FakeOcrEngine(text="лучше 1500", conf=0.97), "none", 400)
    assert better.pages[1].text == "лучше 1500" and better.pages[0].text == doc.pages[0].text
    assert better.status == "ok"
