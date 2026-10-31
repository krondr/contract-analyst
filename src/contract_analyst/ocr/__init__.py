"""OCR engines, engine factory and the per-page OCR routine (rotation + preprocessing)."""

from __future__ import annotations

import numpy as np

from contract_analyst.ocr.base import OcrEngine, OcrResult
from contract_analyst.ocr.preprocess import preprocess

UPRIGHT_CONFIDENCE = 0.6


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


def ocr_page(image: np.ndarray, engine: OcrEngine, mode: str) -> OcrResult:
    prepared = preprocess(image, mode)
    result = engine.recognize(prepared)
    if result.mean_conf >= UPRIGHT_CONFIDENCE:
        return result
    best = result
    for k in (1, 2, 3):
        candidate = engine.recognize(np.rot90(prepared, k=k))
        if candidate.mean_conf > best.mean_conf:
            best = candidate.model_copy(update={"rotation": k * 90})
    return best


__all__ = ["OcrEngine", "OcrResult", "detect_rotation", "get_ocr_engine", "ocr_page", "preprocess"]
