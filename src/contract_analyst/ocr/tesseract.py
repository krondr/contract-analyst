"""Tesseract engine with word-level confidence."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pytesseract
from pytesseract import Output

from contract_analyst.models import OcrWord
from contract_analyst.ocr.base import OcrResult


class TesseractEngine:
    name = "tesseract"

    def __init__(self, lang: str = "rus", psm: int = 6):
        self.lang, self.psm = lang, psm

    def recognize(self, image: np.ndarray) -> OcrResult:
        data = pytesseract.image_to_data(image, lang=self.lang, config=f"--psm {self.psm}", output_type=Output.DICT)
        lines: dict[tuple[int, int, int], list[str]] = defaultdict(list)
        words: list[OcrWord] = []
        for i, raw in enumerate(data["text"]):
            conf = float(data["conf"][i])
            if conf < 0 or not raw.strip():
                continue
            words.append(OcrWord(text=raw, conf=conf / 100))
            lines[(data["block_num"][i], data["par_num"][i], data["line_num"][i])].append(raw)
        text = "\n".join(" ".join(ws) for _, ws in sorted(lines.items()))
        mean = sum(w.conf for w in words) / len(words) if words else 0.0
        return OcrResult(text=text, words=words, mean_conf=mean)
