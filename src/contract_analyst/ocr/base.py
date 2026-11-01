"""OCR engine protocol. Images are RGB (or single-channel) uint8 numpy arrays."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from pydantic import BaseModel

from contract_analyst.models import OcrWord


class OcrResult(BaseModel):
    text: str
    words: list[OcrWord]
    mean_conf: float
    rotation: int = 0  # degrees counter-clockwise applied before recognition
    tall_box_share: float | None = None  # share of detected text boxes taller than wide; None: engine gives no boxes


class OcrEngine(Protocol):
    name: str

    def recognize(self, image: np.ndarray) -> OcrResult: ...
