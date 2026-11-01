"""OCR engines, engine factory and the per-page OCR routine (orientation + preprocessing)."""

from __future__ import annotations

import numpy as np

from contract_analyst.ocr import orientation
from contract_analyst.ocr.base import OcrEngine, OcrResult
from contract_analyst.ocr.preprocess import preprocess

UPRIGHT_CONFIDENCE = 0.6  # engines without text boxes: below this mean confidence the other turns are tried
TALL_BOX_SHARE = 0.5      # more text boxes taller than wide than this: the page lies on its side


def get_ocr_engine(name: str) -> OcrEngine:
    if name == "tesseract":
        from contract_analyst.ocr.tesseract import TesseractEngine

        return TesseractEngine()
    if name == "rapidocr":
        from contract_analyst.ocr.rapid import RapidOcrEngine

        return RapidOcrEngine()
    raise ValueError(f"unknown OCR engine: {name}")


def detect_rotation(image: np.ndarray, engine: OcrEngine) -> int:
    """Counter-clockwise rotation (0/90/180/270) that gives the most confident recognition."""
    scores = {k * 90: engine.recognize(np.rot90(image, k=k)).mean_conf for k in range(4)}
    return max(scores, key=scores.get)


def _turned(engine: OcrEngine, image: np.ndarray, turn: int) -> OcrResult:
    """Recognize the page turned `turn` degrees counter-clockwise; the turn is recorded in the result."""
    if turn % 360 == 0:
        return engine.recognize(image)
    result = engine.recognize(np.ascontiguousarray(np.rot90(image, k=turn // 90)))
    return result.model_copy(update={"rotation": turn % 360})


def ocr_page(image: np.ndarray, engine: OcrEngine, mode: str) -> OcrResult:
    """OCR one page the right way up. Evidence, in order: Tesseract OSD when it is installed and sure; then the text
    boxes of the engine (if most are taller than wide the text still stands on its side, so 90 and 270 degrees more
    are tried and the more confident reading is kept); engines that report no boxes fall back to confidence."""
    prepared = preprocess(image, mode)
    turn = orientation.osd_rotation(prepared)
    result = _turned(engine, prepared, turn or 0)
    if result.tall_box_share is not None:
        if result.tall_box_share <= TALL_BOX_SHARE:
            return result
        candidates = [_turned(engine, prepared, (turn or 0) + extra) for extra in (90, 270)]
        return max(candidates, key=lambda r: r.mean_conf)
    if turn is not None or result.mean_conf >= UPRIGHT_CONFIDENCE:
        return result
    best = result
    for extra in (90, 180, 270):
        candidate = _turned(engine, prepared, extra)
        if candidate.mean_conf > best.mean_conf:
            best = candidate
    return best


__all__ = ["OcrEngine", "OcrResult", "detect_rotation", "get_ocr_engine", "ocr_page", "orientation", "preprocess"]
