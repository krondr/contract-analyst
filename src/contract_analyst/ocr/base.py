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


class OcrEngine(Protocol):
    name: str

    def recognize(self, image: np.ndarray) -> OcrResult: ...
