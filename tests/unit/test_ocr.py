import shutil

import numpy as np
import pymupdf
import pytest

from contract_analyst.eval.text_metrics import cer, wer
from contract_analyst.ocr import get_ocr_engine, ocr_page
from contract_analyst.ocr.preprocess import preprocess
from contract_analyst.ocr.render import render_page
from tests.fixtures.builders import make_scan_pdf, make_text_pdf, render_text_image

TEXT = "Договор поставки № D990000010-01\nг. Тверь 19.08.2020 г.\nЦена товара составляет 1500 рублей."
needs_tesseract = pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract binary missing")


def test_cer_and_wer_count_edits_ignoring_line_breaks():
    assert cer("абв\nгд", "абв гд") == 0.0
    assert cer("абв", "абг") == pytest.approx(1 / 3)
    assert wer("цена 1500 рублей", "цена 1600 рублей") == pytest.approx(1 / 3)


def test_fixture_pdf_text_roundtrip(tmp_path):
    p = make_text_pdf(tmp_path / "a.pdf", [TEXT])
    assert "D990000010-01" in pymupdf.open(p)[0].get_text()


def test_render_page_returns_rgb_image(tmp_path):
    p = make_scan_pdf(tmp_path / "s.pdf", [TEXT])
    img = render_page(p, 1, dpi=100)
    assert img.ndim == 3 and img.shape[2] == 3 and img.dtype == np.uint8


@pytest.mark.ocr
@needs_tesseract
def test_tesseract_reads_clean_cyrillic():
    r = get_ocr_engine("tesseract").recognize(render_text_image(TEXT))
    assert cer(r.text, TEXT) < 0.05 and r.words and all(0 <= w.conf <= 1 for w in r.words)


@pytest.mark.ocr
def test_rapidocr_reads_clean_cyrillic():
    r = get_ocr_engine("rapidocr").recognize(render_text_image(TEXT))
    assert cer(r.text, TEXT) < 0.10 and all(0 <= w.conf <= 1 for w in r.words)


@pytest.mark.ocr
@needs_tesseract
def test_rotation_detected_and_fixed():
    r = ocr_page(np.rot90(render_text_image(TEXT)), get_ocr_engine("tesseract"), "none")
    assert r.rotation in (90, 270) and "1500" in r.text


@pytest.mark.ocr
@needs_tesseract
def test_upright_page_is_not_rotated():
    assert ocr_page(render_text_image(TEXT), get_ocr_engine("tesseract"), "none").rotation == 0


def test_preprocess_binarizes_and_none_is_identity():
    img = render_text_image(TEXT)
    assert set(np.unique(preprocess(img, "deskew_sauvola"))) <= {0, 255}
    assert preprocess(img, "none") is img
