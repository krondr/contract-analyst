"""Optional image preprocessing before OCR: deskew plus Sauvola binarization."""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np

_MAX_SKEW = 10.0


def _to_gray(image: np.ndarray) -> np.ndarray:
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def _deskew(gray: np.ndarray) -> np.ndarray:
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(ink)
    if coords is None:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    if angle > 45:
        angle -= 90
    if abs(angle) < 0.1 or abs(angle) > _MAX_SKEW:
        return gray
    h, w = gray.shape
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, matrix, (w, h), flags=cv2.INTER_CUBIC, borderValue=255)


def preprocess(image: np.ndarray, mode: Literal["none", "deskew_sauvola"]) -> np.ndarray:
    if mode == "none":
        return image
    gray = _deskew(_to_gray(image))
    return cv2.ximgproc.niBlackThreshold(gray, 255, cv2.THRESH_BINARY, 31, 0.2,
                                         binarizationMethod=cv2.ximgproc.BINARIZATION_SAUVOLA)
