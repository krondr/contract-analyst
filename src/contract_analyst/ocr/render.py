"""Rasterize PDF pages for OCR."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pymupdf


def render_page(pdf_path: Path, page_number: int, dpi: int = 300) -> np.ndarray:
    """Render a 1-based page as an RGB uint8 image."""
    with pymupdf.open(pdf_path) as doc:
        pix = doc[page_number - 1].get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
        return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3).copy()
