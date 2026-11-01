import shutil
from types import SimpleNamespace

import numpy as np
import pymupdf
import pytest

from contract_analyst.eval.text_metrics import cer, wer
from contract_analyst.models import OcrWord
from contract_analyst.ocr import get_ocr_engine, ocr_page, orientation
from contract_analyst.ocr.base import OcrResult
from contract_analyst.ocr.preprocess import preprocess
from contract_analyst.ocr.rapid import RapidOcrEngine, reading_order
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


# --- page orientation: Tesseract OSD first, then the geometry of the engine's text boxes -------------------------

PAGE_TEXT = "\n".join(f"{i}. Стороны договорились о поставке товара по цене 1500 рублей за единицу." for i in range(1, 25))


def test_osd_turn_is_the_counter_clockwise_fix(monkeypatch):
    answers = iter([{"rotate": 90, "orientation_conf": 9.9}, {"rotate": 0, "orientation_conf": 25.0},
                    {"rotate": 270, "orientation_conf": 0.5}])
    monkeypatch.setattr(orientation.pytesseract, "image_to_osd", lambda *args, **kwargs: next(answers))
    page = np.zeros((10, 10), np.uint8)
    assert orientation.osd_rotation(page) == 270  # "rotate 90 clockwise" is a 270 degree counter-clockwise turn
    assert orientation.osd_rotation(page) == 0
    assert orientation.osd_rotation(page) is None  # OSD is not sure


def test_osd_turn_is_none_when_tesseract_cannot_tell(monkeypatch):
    def fail(*args, **kwargs):
        raise orientation.pytesseract.TesseractError(1, "Too few characters. Skipping this page")

    monkeypatch.setattr(orientation.pytesseract, "image_to_osd", fail)
    assert orientation.osd_rotation(np.zeros((10, 10), np.uint8)) is None


class _MarkerEngine:
    """Fake engine for a portrait page with a dark top-left pixel: it reads well only when the page is upright and,
    like a text detector, reports tall boxes while the page lies on its side."""

    name = "marker"

    def __init__(self, boxes: bool = True):
        self.boxes, self.calls = boxes, 0

    def recognize(self, image: np.ndarray) -> OcrResult:
        self.calls += 1
        sideways = image.shape[0] < image.shape[1]
        conf = 0.95 if not sideways and image[0, 0] == 0 else 0.30
        return OcrResult(text="текст", words=[OcrWord(text="текст", conf=conf)], mean_conf=conf,
                         tall_box_share=(1.0 if sideways else 0.0) if self.boxes else None)


def _portrait_page() -> np.ndarray:
    page = np.full((60, 40), 255, np.uint8)
    page[0, 0] = 0
    return page


def test_osd_decides_the_turn(monkeypatch):
    monkeypatch.setattr(orientation, "osd_rotation", lambda image: 270)
    engine = _MarkerEngine()
    result = ocr_page(np.rot90(_portrait_page()), engine, "none")  # page turned 90 degrees counter-clockwise
    assert (result.rotation, result.mean_conf, engine.calls) == (270, 0.95, 1)


def test_box_geometry_turns_a_sideways_page_without_osd(monkeypatch):
    monkeypatch.setattr(orientation, "osd_rotation", lambda image: None)
    result = ocr_page(np.rot90(_portrait_page()), _MarkerEngine(), "none")
    assert (result.rotation, result.mean_conf) == (270, 0.95)  # 90 and 270 tried, the better one kept


def test_box_geometry_corrects_osd_that_left_text_on_its_side(monkeypatch):
    monkeypatch.setattr(orientation, "osd_rotation", lambda image: 0)
    result = ocr_page(np.rot90(_portrait_page()), _MarkerEngine(), "none")
    assert (result.rotation, result.mean_conf) == (270, 0.95)


def test_upright_page_with_wide_boxes_is_read_once(monkeypatch):
    monkeypatch.setattr(orientation, "osd_rotation", lambda image: None)
    engine = _MarkerEngine()
    assert (ocr_page(_portrait_page(), engine, "none").rotation, engine.calls) == (0, 1)


class _FaintSidewaysEngine:
    """RapidOCR on Linux for a faint sideways scan (CI): no text box at all one way up; the other way up tall boxes
    that it still reads through rotated crops, more confidently than the upright page; upright, flat confident lines;
    upside down, flat lines it cannot read."""

    name = "faint"

    def recognize(self, image: np.ndarray) -> OcrResult:
        def read(conf: float, tall: float) -> OcrResult:
            return OcrResult(text="текст", words=[OcrWord(text="текст", conf=conf)], mean_conf=conf, tall_box_share=tall)

        nothing = OcrResult(text="", words=[], mean_conf=0.0, tall_box_share=0.0)
        if image.shape[0] < image.shape[1]:  # on its side
            return nothing if image[-1, 0] == 0 else read(0.987, 1.0)
        return read(0.983, 0.0) if image[0, 0] == 0 else nothing


def test_page_where_no_text_is_found_is_not_taken_as_upright(monkeypatch):
    monkeypatch.setattr(orientation, "osd_rotation", lambda image: None)
    result = ocr_page(np.ascontiguousarray(np.rot90(_portrait_page())), _FaintSidewaysEngine(), "none")
    assert (result.rotation, result.mean_conf) == (270, 0.983)


def test_blank_page_keeps_its_turn(monkeypatch):
    monkeypatch.setattr(orientation, "osd_rotation", lambda image: None)
    blank = SimpleNamespace(name="blank", recognize=lambda image: OcrResult(text="", words=[], mean_conf=0.0,
                                                                            tall_box_share=0.0))
    assert ocr_page(_portrait_page(), blank, "none").rotation == 0


def test_engine_without_boxes_falls_back_to_confidence(monkeypatch):
    monkeypatch.setattr(orientation, "osd_rotation", lambda image: None)
    result = ocr_page(np.rot90(_portrait_page()), _MarkerEngine(boxes=False), "none")
    assert (result.rotation, result.mean_conf) == (270, 0.95)


@pytest.mark.ocr
@needs_tesseract
def test_osd_finds_a_quarter_turn_on_a_rendered_page():
    page = render_text_image(PAGE_TEXT)
    assert orientation.osd_rotation(page) == 0
    assert orientation.osd_rotation(np.ascontiguousarray(np.rot90(page))) == 270
    assert orientation.osd_rotation(np.ascontiguousarray(np.rot90(page, k=3))) == 90


@pytest.mark.ocr
@pytest.mark.parametrize("use_osd", [True, False])
def test_degraded_sideways_scan_is_turned_upright_for_rapidocr(tmp_path, monkeypatch, use_osd):
    """Regression for E1: `ocr_page` used to leave RapidOCR pages on their side (its confidence stays high)."""
    from contract_analyst.eval.ocr_bench import make_synthetic_scans

    if use_osd and shutil.which("tesseract") is None:
        pytest.skip("tesseract binary missing")
    if not use_osd:
        monkeypatch.setattr(orientation, "osd_rotation", lambda image: None)
    pdf = make_text_pdf(tmp_path / "a.pdf", [PAGE_TEXT] * 4)
    (scan,) = [s for s in make_synthetic_scans([pdf], n_pages=4, seed=3) if s.meta["rot90"]]
    result = ocr_page(scan.image, get_ocr_engine("rapidocr"), "none")
    assert (scan.meta["rot90"] + result.rotation) % 360 == 0 and cer(result.text, scan.truth) < 0.15


# --- RapidOCR reading order --------------------------------------------------------------------------------------


def _quad(x: float, y: float, width: float, height: float, skew_deg: float) -> np.ndarray:
    """Corners (top-left, top-right, bottom-right, bottom-left) of a text box on a page skewed by `skew_deg`."""
    a = np.deg2rad(skew_deg)
    along, across = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
    top_left = np.array([x, y])
    return np.array([top_left, top_left + width * along, top_left + width * along + height * across,
                     top_left + height * across])


def _skewed_lines(n_lines: int, skew_deg: float = 3.0) -> np.ndarray:
    """Lines 45 px apart on a page skewed by 3 degrees, each cut by the detector into two fragments; listed line by
    line, left to right. The right fragment starts about 40 px lower than the left one, so sorting by the top-left
    corner in 10 px buckets put it after the next line."""
    a = np.deg2rad(skew_deg)
    boxes = []
    for line in range(n_lines):
        x, y = 100.0, 100.0 + 45 * line
        boxes += [_quad(x, y, 700, 30, skew_deg), _quad(x + 760 * np.cos(a), y + 760 * np.sin(a), 700, 30, skew_deg)]
    return np.array(boxes)


def test_reading_order_keeps_fragments_of_a_skewed_line_together():
    shuffled = [4, 1, 5, 0, 3, 2]
    order = reading_order(_skewed_lines(3)[shuffled])
    assert [[shuffled[i] for i in line] for line in order] == [[0, 1], [2, 3], [4, 5]]
    assert reading_order(np.zeros((0, 4, 2))) == []


def test_rapid_engine_text_follows_reading_order():
    boxes = _skewed_lines(2)
    texts = ("Цена товара", "1500 рублей.", "Срок поставки", "30 дней.")
    engine = RapidOcrEngine()
    engine._engine = lambda image: SimpleNamespace(boxes=boxes[::-1], txts=texts[::-1], scores=(0.9, 0.8, 0.7, 0.6))
    result = engine.recognize(np.zeros((20, 20, 3), np.uint8))
    assert result.text == "Цена товара 1500 рублей.\nСрок поставки 30 дней."
    assert [(w.text, w.conf) for w in result.words[:3]] == [("Цена", 0.6), ("товара", 0.6), ("1500", 0.7)]
    assert result.tall_box_share == 0.0


def test_rapid_engine_keeps_unreadable_boxes_as_orientation_evidence():
    """A page on its side is read as noise: the text is dropped, but its tall boxes still tell `ocr_page` to turn it."""
    standing = np.array([_quad(100, 100, 30, 600, 0.0), _quad(150, 100, 30, 600, 0.0)])
    engine = RapidOcrEngine()
    engine._engine = lambda image: SimpleNamespace(boxes=standing, txts=("ыъ", "ьъ"), scores=(0.2, 0.3))
    result = engine.recognize(np.zeros((20, 20, 3), np.uint8))
    assert (result.text, result.words, result.mean_conf, result.tall_box_share) == ("", [], 0.0, 1.0)
