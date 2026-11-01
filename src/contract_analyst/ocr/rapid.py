"""PaddleOCR PP-OCRv5 (East-Slavic recognizer) through RapidOCR/ONNX; confidence is per detected text box."""

from __future__ import annotations

import cv2
import numpy as np

from contract_analyst.models import OcrWord
from contract_analyst.ocr.base import OcrResult

MIN_LINE_OVERLAP = 0.5  # share of the lower box height two boxes must overlap vertically to be on one line
MIN_TEXT_SCORE = 0.5    # RapidOCR's own default: lines read with less confidence are dropped from the text


def reading_order(boxes: np.ndarray) -> list[list[int]]:
    """Box indices grouped into text lines, lines top to bottom and boxes in a line left to right.

    Boxes are quadrilaterals (top-left, top-right, bottom-right, bottom-left). The page skew, the median angle of the
    top edges of wide boxes, is undone first; then a box joins a line when it overlaps the line vertically by at least
    MIN_LINE_OVERLAP of the lower height. So the pieces of one skewed line stay together, even when the right piece
    starts lower than the next line's left piece.
    """
    pts = np.asarray(boxes, dtype=np.float64).reshape(-1, 4, 2)
    if len(pts) == 0:
        return []
    top_edge = pts[:, 1] - pts[:, 0]
    wide = np.ptp(pts[:, :, 0], axis=1) >= np.ptp(pts[:, :, 1], axis=1)
    skew = float(np.median(np.arctan2(top_edge[wide, 1], top_edge[wide, 0]))) if wide.any() else 0.0
    cos, sin = np.cos(skew), np.sin(skew)
    xs = pts[:, :, 0] * cos + pts[:, :, 1] * sin  # coordinates on the page turned back by the skew
    ys = pts[:, :, 1] * cos - pts[:, :, 0] * sin
    top, bottom, left = ys.min(axis=1), ys.max(axis=1), xs.min(axis=1)

    lines: list[dict] = []
    for i in np.argsort((top + bottom) / 2, kind="stable"):
        for line in reversed(lines[-3:]):  # boxes come by height, so the line is one of the last ones
            line_top, line_bottom = line["top"] / len(line["boxes"]), line["bottom"] / len(line["boxes"])
            overlap = min(bottom[i], line_bottom) - max(top[i], line_top)
            if overlap >= MIN_LINE_OVERLAP * min(bottom[i] - top[i], line_bottom - line_top):
                line["boxes"].append(int(i))
                line["top"] += top[i]
                line["bottom"] += bottom[i]
                break
        else:
            lines.append({"boxes": [int(i)], "top": top[i], "bottom": bottom[i]})
    lines.sort(key=lambda line: (line["top"] + line["bottom"]) / len(line["boxes"]))
    return [sorted(line["boxes"], key=lambda i: left[i]) for line in lines]


def _tall_share(boxes: np.ndarray) -> float:
    pts = np.asarray(boxes, dtype=np.float64).reshape(-1, 4, 2)
    return float(np.mean(np.ptp(pts[:, :, 1], axis=1) > np.ptp(pts[:, :, 0], axis=1))) if len(pts) else 0.0


class RapidOcrEngine:
    name = "rapidocr"

    def __init__(self) -> None:
        self._engine = None

    def _get(self):
        if self._engine is None:
            from rapidocr import RapidOCR
            from rapidocr.utils.typings import LangRec, ModelType, OCRVersion

            # No text-line angle classifier: `ocr_page` sets the page orientation, and the classifier (trained on
            # Chinese) flipped upright Cyrillic lines and made upside-down pages read as confidently as upright ones.
            # All boxes come back (text_score 0): the weak ones are dropped from the text here, but still count as
            # orientation evidence.
            self._engine = RapidOCR(params={
                "Global.log_level": "error",
                "Global.use_cls": False,
                "Global.text_score": 0.0,
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
        if out.boxes is None or out.txts is None or len(out.txts) == 0:
            return OcrResult(text="", words=[], mean_conf=0.0, tall_box_share=0.0)
        boxes = np.asarray(out.boxes, dtype=np.float64).reshape(-1, 4, 2)
        keep = [i for i, (txt, score) in enumerate(zip(out.txts, out.scores, strict=False))
                if float(score) >= MIN_TEXT_SCORE and txt.strip()]
        tall = _tall_share(boxes)
        if not keep:
            return OcrResult(text="", words=[], mean_conf=0.0, tall_box_share=tall)
        txts, scores = [out.txts[i] for i in keep], [float(out.scores[i]) for i in keep]
        lines = reading_order(boxes[keep])
        words = [OcrWord(text=w, conf=scores[i]) for line in lines for i in line for w in txts[i].split()]
        text = "\n".join(" ".join(txts[i] for i in line) for line in lines)
        return OcrResult(text=text, words=words, mean_conf=float(np.mean(scores)), tall_box_share=tall)
