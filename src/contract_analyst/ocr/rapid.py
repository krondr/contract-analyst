"""PaddleOCR PP-OCRv5 (East-Slavic recognizer) through RapidOCR/ONNX; confidence is per text line."""

from __future__ import annotations

import cv2
import numpy as np

from contract_analyst.models import OcrWord
from contract_analyst.ocr.base import OcrResult


class RapidOcrEngine:
    name = "rapidocr"

    def __init__(self) -> None:
        self._engine = None

    def _get(self):
        if self._engine is None:
            from rapidocr import RapidOCR
            from rapidocr.utils.typings import LangRec, ModelType, OCRVersion

            self._engine = RapidOCR(params={
                "Global.log_level": "error",
                "Rec.lang_type": LangRec.ESLAV,
                "Rec.ocr_version": OCRVersion.PPOCRV5,
                "Rec.model_type": ModelType.MOBILE,
                "Det.ocr_version": OCRVersion.PPOCRV5,
                "Det.model_type": ModelType.MOBILE,
            })
        return self._engine

    def recognize(self, image: np.ndarray) -> OcrResult:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if image.ndim == 2 else image
        out = self._get()(rgb)
        if out.txts is None or len(out.txts) == 0:
            return OcrResult(text="", words=[], mean_conf=0.0)
        items = sorted(zip(out.boxes, out.txts, out.scores), key=lambda it: (round(float(it[0][0][1]) / 10), float(it[0][0][0])))
        words = [OcrWord(text=w, conf=float(score)) for _, txt, score in items for w in txt.split()]
        text = "\n".join(txt for _, txt, _ in items)
        return OcrResult(text=text, words=words, mean_conf=float(np.mean([s for *_, s in items])))
