"""Page orientation from Tesseract OSD (orientation and script detection), independent of the OCR engine."""

from __future__ import annotations

import numpy as np
import pytesseract

# OSD reports how sure it is; on the E1 synthetic scans correct answers on upright pages scored 2.45 and more.
# Below this the answer is not used and `ocr_page` falls back to the geometry of the text boxes.
OSD_MIN_CONFIDENCE = 2.0


def osd_rotation(image: np.ndarray, min_confidence: float = OSD_MIN_CONFIDENCE) -> int | None:
    """Counter-clockwise turn (0, 90, 180 or 270 degrees) that puts the page upright, or None when Tesseract is not
    installed, OSD fails (too little text) or is not sure."""
    try:
        osd = pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT)
        rotate, confidence = int(osd["rotate"]), float(osd["orientation_conf"])
    except (OSError, RuntimeError, ValueError, KeyError):  # binary or OSD data missing, too few characters
        return None
    if confidence < min_confidence:
        return None
    return (360 - rotate) % 360  # OSD gives the clockwise turn that fixes the page
