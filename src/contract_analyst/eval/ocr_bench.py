"""E1: choose the OCR engine and preprocessing by measurement.

Synthetic scans are text-layer PDF pages rendered at 200 dpi and degraded the way a scanner would; their truth is
the page's own (normalized) text layer. Hand-typed real scanned pages from the golden set are scored too when
present. Critical tokens are document numbers and numbers (dates, amounts, clause numbers): for every engine x
preprocessing variant we find the smallest word-confidence threshold at which >= 99% of the accepted critical
tokens are correct, and which share of the true critical tokens is then accepted correctly (coverage).

Run: uv run python -m contract_analyst.eval.ocr_bench --data data/raw --out docs/experiments/e1_ocr.md
The output holds metrics only: no document text, numbers or file names.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import os
import platform
import re
import subprocess
import sys
import time
import unicodedata
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pymupdf

from contract_analyst.eval.golden import load_golden
from contract_analyst.eval.text_metrics import cer, wer
from contract_analyst.ingest.normalize import normalize_text
from contract_analyst.models import OcrWord
from contract_analyst.ocr import OcrEngine, get_ocr_engine, ocr_page
from contract_analyst.ocr.render import render_page
from contract_analyst.parsing.patterns import DOC_NUMBER_RE

ENGINES = ("tesseract", "rapidocr")
MODES = ("none", "deskew_sauvola")

SYNTH_DPI = 200
MIN_TEXT_CHARS = 50     # the reader's rule: fewer non-space characters means no usable text layer
MAX_IMAGE_SHARE = 0.01  # pages with bigger images show content that is missing from the text layer
ROTATED_SHARE = 0.25
SKEW_DEG = 3.0
BLUR_SIGMA = (0.0, 1.2)
NOISE_SIGMA = (0.0, 12.0)
JPEG_QUALITY = (40, 90)
SCALE = (0.6, 1.0)

TARGET_PRECISION = 0.99
GRID = tuple(round(0.50 + i / 100, 2) for i in range(50))  # 0.50 .. 0.99
THRESHOLD_MARGIN = 0.05  # spec 6.14: the config threshold keeps a margin over the calibrated one
MAX_THRESHOLD = 0.99
TIE = 0.01
_EPS = 1e-9

CRITICAL_RE = re.compile(rf"(?:{DOC_NUMBER_RE.pattern})|\d[\d.,]*\d|\d")  # matched against whole words
_EDGES_RE = re.compile(r"^[\W_]+|[\W_]+$")  # punctuation and symbols around a word: «19», (1500), 3.1;, №1
_ILLEGIBLE = "[неразборчиво]"

Log = Callable[[str], None]


@dataclass(frozen=True, eq=False)
class SyntheticScan:
    image: np.ndarray  # RGB uint8
    truth: str         # normalized text
    meta: dict         # degradation parameters and source indices; never text or file names


@dataclass(frozen=True)
class CurvePoint:
    t: float
    precision: float | None  # None when nothing is accepted
    coverage: float
    accepted: int
    correct: int


@dataclass(frozen=True)
class BenchRow:
    engine: str
    mode: str
    cer: float
    wer: float
    crit_coverage_at_p99: float
    threshold: float | None  # smallest grid t with precision >= TARGET_PRECISION; None if none reaches it
    sec_per_page: float
    crit_precision: float | None = None  # precision at `threshold`
    rec_coverage: float | None = None    # coverage and precision at `recommended_threshold`
    rec_precision: float | None = None
    n_pages: int = 0
    n_crit_truth: int = 0
    n_doc_numbers: int = 0
    doc_number_recall: float | None = None  # share of true doc numbers read exactly, at any confidence
    cer_rotated: float | None = None        # CER on the pages turned by 90 degrees
    rotation_fixed: float | None = None     # share of those pages that `ocr_page` turned back upright
    wer_unordered: float | None = None      # share of reference words missing from the OCR text, order ignored
    real_cer: float | None = None
    real_wer: float | None = None
    real_crit_precision: float | None = None  # real pages, at `recommended_threshold`
    real_crit_coverage: float | None = None
    n_real_pages: int = 0
    load_avg: float | None = None  # mean 1-minute load average of the machine while this variant ran
    curve: tuple[CurvePoint, ...] = field(default=(), repr=False)

    @property
    def recommended_threshold(self) -> float | None:
        return recommended_threshold(self.threshold)


def recommended_threshold(t: float | None) -> float | None:
    return None if t is None else min(round(t + THRESHOLD_MARGIN, 2), MAX_THRESHOLD)


# --- critical tokens and the precision/coverage curve ----------------------------------------------------------


def critical_token(word: str) -> str | None:
    """The word without edge punctuation if it is a document number (D + 9 digits, optional -NN) or a number
    (dates and amounts included). A mangled word such as a doc number read with a Cyrillic letter is not a
    critical token: the parsers would not take it for a value, so it costs coverage, not precision."""
    core = _EDGES_RE.sub("", word)
    return core if CRITICAL_RE.fullmatch(core) else None


def critical_tokens(text: str) -> list[str]:
    """Critical tokens of a text, in order (words are split on whitespace)."""
    return [t for t in map(critical_token, text.split()) if t is not None]


def critical_words(words: Sequence[OcrWord]) -> list[tuple[str, float]]:
    """Critical tokens of OCR words, each with its word's confidence."""
    return [(t, w.conf) for w in words if (t := critical_token(w.text)) is not None]


def _is_doc_number(token: str) -> bool:
    return token.startswith("D")


PageTokens = tuple[Sequence[tuple[str, float]], Counter]  # (OCR tokens with confidence, truth token counts)


def _point(pages: Sequence[PageTokens], t: float) -> CurvePoint:
    accepted = correct = 0
    for ocr_tokens, truth in pages:
        taken = Counter(token for token, conf in ocr_tokens if conf >= t - _EPS)
        accepted += taken.total()
        correct += sum(min(n, truth[token]) for token, n in taken.items())  # each truth token matches once
    total = sum(truth.total() for _, truth in pages)
    return CurvePoint(t=t, precision=correct / accepted if accepted else None,
                      coverage=correct / total if total else 0.0, accepted=accepted, correct=correct)


def crit_curve(pages: Sequence[PageTokens], grid: Sequence[float] = GRID) -> list[CurvePoint]:
    """Micro-averaged precision and coverage of accepted critical tokens for every threshold of the grid."""
    return [_point(pages, t) for t in grid]


def threshold_at_precision(curve: Sequence[CurvePoint],
                           target: float = TARGET_PRECISION) -> tuple[float | None, float]:
    """Smallest threshold whose precision reaches the target, and the coverage there; (None, 0.0) if none does."""
    for p in curve:
        if p.precision is not None and p.precision >= target - _EPS:
            return p.t, p.coverage
    return None, 0.0


# --- synthetic scans ---------------------------------------------------------------------------------------------


def _chars(text: str) -> int:
    return sum(1 for ch in text if not ch.isspace())


def _unrotated_bounds(page: pymupdf.Page) -> pymupdf.Rect:
    """The page area in the unrotated coordinates that word and image boxes use."""
    return page.rect * page.derotation_matrix


def _image_rects(page: pymupdf.Page) -> list[pymupdf.Rect]:
    bounds = _unrotated_bounds(page)
    rects = [pymupdf.Rect(info["bbox"]) & bounds for info in page.get_image_info()]
    return [r for r in rects if not r.is_empty]


def _eligible(page: pymupdf.Page) -> bool:
    if _chars(normalize_text(page.get_text())) < MIN_TEXT_CHARS:
        return False
    image_area = sum(r.width * r.height for r in _image_rects(page))
    return image_area <= MAX_IMAGE_SHARE * page.rect.width * page.rect.height


def _page_truth(page: pymupdf.Page, masks: Sequence[pymupdf.Rect]) -> str:
    """Normalized text layer in reading order, without words off the page or under masked (painted-over) images."""
    bounds = _unrotated_bounds(page)
    lines: dict[tuple[int, int], list[str]] = {}
    for x0, y0, x1, y1, word, block, line, _ in page.get_text("words", sort=True):
        center = pymupdf.Point((x0 + x1) / 2, (y0 + y1) / 2)
        if center in bounds and not any(center in r for r in masks):
            lines.setdefault((block, line), []).append(word)
    return normalize_text("\n".join(" ".join(words) for words in lines.values()))


def _paint_white(image: np.ndarray, page: pymupdf.Page, masks: Sequence[pymupdf.Rect], dpi: int) -> np.ndarray:
    zoom = dpi / 72
    for rect in masks:
        r = rect * page.rotation_matrix
        x0, y0 = max(0, int(r.x0 * zoom)), max(0, int(r.y0 * zoom))
        image[y0:int(np.ceil(r.y1 * zoom)), x0:int(np.ceil(r.x1 * zoom))] = 255
    return image


def _rotate_bound(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate by `angle` degrees counter-clockwise on a canvas large enough to keep the whole page."""
    h, w = image.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos + 0.5), int(h * cos + w * sin + 0.5)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(image, m, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))


def _degrade(image: np.ndarray, p: dict, rng: np.random.Generator) -> np.ndarray:
    """Scanner model: page orientation and skew, optics blur, sampling at a lower resolution, sensor noise,
    JPEG on save; finally the page is resampled back to the render size, as the reader rasterizes a scanned
    PDF page at a fixed dpi whatever the resolution of the embedded image."""
    out = np.rot90(image, k=p["rot90"] // 90) if p["rot90"] else image
    out = _rotate_bound(np.ascontiguousarray(out), p["skew_deg"])
    if p["blur_sigma"] > 0:
        out = cv2.GaussianBlur(out, (0, 0), p["blur_sigma"])
    h, w = out.shape[:2]
    if p["scale"] < 1:
        out = cv2.resize(out, (max(1, round(w * p["scale"])), max(1, round(h * p["scale"]))),
                         interpolation=cv2.INTER_AREA)
    if p["noise_sigma"] > 0:
        noise = rng.normal(0.0, p["noise_sigma"], out.shape[:2]).astype(np.float32)[..., None]
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, p["jpeg_quality"]])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    out = cv2.cvtColor(cv2.imdecode(jpeg, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    if out.shape[:2] != (h, w):
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
    return out


def _select_pages(pdfs: Sequence[Path], n_pages: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    """(pdf index, 1-based page) pairs spread over the documents: shuffled pages, taken one per document per round."""
    pools: list[list[tuple[int, int]]] = []
    for di, pdf in enumerate(pdfs):
        with pymupdf.open(pdf) as doc:
            eligible = [page.number + 1 for page in doc if _eligible(page)]
        pools.append([(di, int(n)) for n in rng.permutation(eligible)])
    order = [pools[int(i)] for i in rng.permutation(len(pools))]
    selected: list[tuple[int, int]] = []
    while len(selected) < n_pages and any(order):
        for pool in order:
            if pool and len(selected) < n_pages:
                selected.append(pool.pop(0))
    return selected


def _draw_params(n: int, rng: np.random.Generator) -> list[dict]:
    rotated = {int(i) for i in rng.choice(n, size=round(n * ROTATED_SHARE), replace=False)} if n else set()
    params = []
    for i in range(n):
        turn = (90 if rng.random() < 0.5 else 270) if i in rotated else 0
        params.append({
            "rot90": turn,
            "skew_deg": round(float(rng.uniform(-SKEW_DEG, SKEW_DEG)), 3),
            "blur_sigma": round(float(rng.uniform(*BLUR_SIGMA)), 3),
            "noise_sigma": round(float(rng.uniform(*NOISE_SIGMA)), 3),
            "jpeg_quality": int(rng.integers(JPEG_QUALITY[0], JPEG_QUALITY[1] + 1)),
            "scale": round(float(rng.uniform(*SCALE)), 3),
        })
    return params


def make_synthetic_scans(text_pdfs: list[Path], n_pages: int = 40, seed: int = 0) -> list[SyntheticScan]:
    """Degraded renders of text-layer pages with their normalized text layer as truth; same seed -> same scans."""
    pdfs = sorted((Path(p) for p in text_pdfs), key=lambda p: unicodedata.normalize("NFC", str(p)))
    rng = np.random.default_rng(seed)
    selected = _select_pages(pdfs, n_pages, rng)
    scans = []
    for i, ((di, page_no), params) in enumerate(zip(selected, _draw_params(len(selected), rng), strict=True)):
        with pymupdf.open(pdfs[di]) as doc:
            page = doc[page_no - 1]
            masks = _image_rects(page)
            truth = _page_truth(page, masks)
            image = _paint_white(render_page(pdfs[di], page_no, dpi=SYNTH_DPI), page, masks, SYNTH_DPI)
        degraded = _degrade(image, params, np.random.default_rng([seed, i]))
        scans.append(SyntheticScan(image=degraded, truth=truth,
                                   meta={"doc_index": di, "page": page_no, "masked_images": len(masks), **params}))
    return scans


# --- real scanned pages (hand-typed transcriptions from the golden set) ------------------------------------------


def _resolve(data_dir: Path, rel: str) -> Path:
    path = data_dir / rel
    if path.exists():
        return path
    wanted = unicodedata.normalize("NFC", rel)
    for candidate in data_dir.rglob("*"):
        if unicodedata.normalize("NFC", str(candidate.relative_to(data_dir))) == wanted:
            return candidate
    raise FileNotFoundError(f"golden scan page refers to a missing file: {rel}")


def load_real_pages(golden_path: Path | None, data_dir: Path, dpi: int = 300) -> list[SyntheticScan]:
    """Real scanned pages with their hand-typed text (`scans` of golden.yaml); [] when there are none yet."""
    if golden_path is None or not Path(golden_path).exists():
        return []
    pages = []
    for i, scan in enumerate(load_golden(golden_path).scans):
        path = _resolve(data_dir, scan.file)
        if path.suffix.lower() == ".pdf":
            image = render_page(path, scan.page, dpi=dpi)
        else:
            bgr = cv2.imread(str(path))
            if bgr is None:
                raise ValueError(f"cannot read golden scan image: {scan.file}")
            image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        truth = normalize_text(scan.text.replace(_ILLEGIBLE, " "))
        pages.append(SyntheticScan(image=image, truth=truth, meta={"source": "real", "index": i, "page": scan.page}))
    return pages


# --- benchmark ---------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _PageRun:
    cer: float
    wer: float
    ref_chars: int
    ref_words: int
    ocr_tokens: list[tuple[str, float]]
    truth_tokens: Counter
    seconds: float
    rotation_ok: bool | None  # None for pages that were not turned by 90 degrees
    words_missed: int         # reference words absent from the OCR text (multisets, order ignored)

    @property
    def rotated(self) -> bool:
        return self.rotation_ok is not None


def _run_page(engine: OcrEngine, mode: str, scan: SyntheticScan) -> _PageRun:
    start = time.perf_counter()
    result = ocr_page(scan.image, engine, mode)
    seconds = time.perf_counter() - start
    hyp = normalize_text(result.text)  # the reader normalizes OCR text the same way
    turned = int(scan.meta.get("rot90") or 0)
    return _PageRun(cer=cer(hyp, scan.truth), wer=wer(hyp, scan.truth),
                    ref_chars=max(1, len(" ".join(scan.truth.split()))), ref_words=max(1, len(scan.truth.split())),
                    ocr_tokens=critical_words(result.words), truth_tokens=Counter(critical_tokens(scan.truth)),
                    seconds=seconds, rotation_ok=(turned + result.rotation) % 360 == 0 if turned else None,
                    words_missed=(Counter(scan.truth.split()) - Counter(hyp.split())).total())


def _micro(runs: Sequence[_PageRun]) -> tuple[float, float]:
    """CER and WER over all pages: per-page rates weighted by the reference length (= total edits / total length)."""
    chars, words = sum(r.ref_chars for r in runs), sum(r.ref_words for r in runs)
    return (sum(r.cer * r.ref_chars for r in runs) / chars, sum(r.wer * r.ref_words for r in runs) / words)


def _doc_number_recall(runs: Sequence[_PageRun]) -> float | None:
    pages = [([(t, c) for t, c in r.ocr_tokens if _is_doc_number(t)],
              Counter({t: n for t, n in r.truth_tokens.items() if _is_doc_number(t)})) for r in runs]
    return _point(pages, 0.0).coverage if any(truth for _, truth in pages) else None


def _row(engine: str, mode: str, synth: Sequence[_PageRun], real: Sequence[_PageRun],
         load_avg: float | None = None) -> BenchRow:
    pages = [(r.ocr_tokens, r.truth_tokens) for r in synth]
    curve = tuple(crit_curve(pages))
    t, coverage = threshold_at_precision(curve)
    rec = recommended_threshold(t)
    at_t = _point(pages, t) if t is not None else None
    at_rec = _point(pages, rec) if rec is not None else None
    cer_s, wer_s = _micro(synth) if synth else (float("nan"), float("nan"))
    rotated = [r for r in synth if r.rotated]
    real_cer = real_wer = real_p = real_c = None
    if real:
        real_cer, real_wer = _micro(real)
        if rec is not None:
            real_point = _point([(r.ocr_tokens, r.truth_tokens) for r in real], rec)
            real_p, real_c = real_point.precision, real_point.coverage
    return BenchRow(
        engine=engine, mode=mode, cer=cer_s, wer=wer_s, crit_coverage_at_p99=coverage, threshold=t,
        sec_per_page=sum(r.seconds for r in synth) / len(synth) if synth else float("nan"),
        crit_precision=at_t.precision if at_t else None,
        rec_coverage=at_rec.coverage if at_rec else None, rec_precision=at_rec.precision if at_rec else None,
        n_pages=len(synth), n_crit_truth=sum(r.truth_tokens.total() for r in synth),
        n_doc_numbers=sum(n for r in synth for tok, n in r.truth_tokens.items() if _is_doc_number(tok)),
        doc_number_recall=_doc_number_recall(synth), cer_rotated=_micro(rotated)[0] if rotated else None,
        rotation_fixed=sum(bool(r.rotation_ok) for r in rotated) / len(rotated) if rotated else None,
        wer_unordered=sum(r.words_missed for r in synth) / sum(r.ref_words for r in synth) if synth else None,
        real_cer=real_cer, real_wer=real_wer, real_crit_precision=real_p, real_crit_coverage=real_c,
        n_real_pages=len(real), load_avg=load_avg, curve=curve)


def _load_1min() -> float | None:
    try:
        return os.getloadavg()[0]
    except OSError:
        return None


def _interleaved(variants: Sequence[tuple[OcrEngine, str]], pages: Sequence[SyntheticScan], label: str,
                 log: Log | None) -> tuple[list[list[_PageRun]], list[list[float]]]:
    """Every page through every variant before the next page, starting with a different variant each time:
    a change in machine load then affects all variants alike, and the speed comparison stays fair."""
    runs: list[list[_PageRun]] = [[] for _ in variants]
    loads: list[list[float]] = [[] for _ in variants]
    for n, page in enumerate(pages):
        shift = n % len(variants)
        for i in [*range(shift, len(variants)), *range(shift)]:
            engine, mode = variants[i]
            runs[i].append(_run_page(engine, mode, page))
            if (load := _load_1min()) is not None:
                loads[i].append(load)
        if log:
            log(f"{label} {n + 1}/{len(pages)}: " + "; ".join(
                f"{e.name}/{m} {r[-1].seconds:.1f}s CER {r[-1].cer:.3f}" for (e, m), r in zip(variants, runs, strict=True)))
    return runs, loads


def bench(engines: Sequence[str | OcrEngine], modes: Sequence[str], scans: Sequence[SyntheticScan],
          real_pages: Sequence[SyntheticScan], log: Log | None = None) -> list[BenchRow]:
    """One row per engine x preprocessing mode; pages go through `ocr_page` exactly as in the reader."""
    variants: list[tuple[OcrEngine, str]] = []
    for spec in engines:
        engine = get_ocr_engine(spec) if isinstance(spec, str) else spec
        if scans:
            engine.recognize(scans[0].image)  # warm-up: model loading is not counted in sec/page
        variants += [(engine, mode) for mode in modes]
    synth, loads = _interleaved(variants, scans, "synthetic", log)
    real, _ = _interleaved(variants, real_pages, "real", log)
    rows = []
    for (engine, mode), runs, real_runs, load in zip(variants, synth, real, loads, strict=True):
        rows.append(_row(engine.name, mode, runs, real_runs, sum(load) / len(load) if load else None))
        if log:
            log(f"[{engine.name}/{mode}] done: {_row_summary(rows[-1])}")
    return rows


def select_winner(rows: Sequence[BenchRow]) -> BenchRow:
    """Pre-registered rule: 1) max critical coverage at precision >= 0.99; 2) within TIE of the best -> lower CER;
    3) then lower seconds per page."""
    if not rows:
        raise ValueError("no benchmark rows to choose from")
    best = max(r.crit_coverage_at_p99 for r in rows)
    tied = [r for r in rows if best - r.crit_coverage_at_p99 <= TIE + _EPS]
    return min(tied, key=lambda r: (r.cer, r.sec_per_page))


def _row_summary(r: BenchRow) -> str:
    return (f"CER {r.cer:.4f}, WER {r.wer:.4f}, coverage@p99 {r.crit_coverage_at_p99:.3f}, t {_f(r.threshold, 2)}, "
            f"{r.sec_per_page:.2f} s/page")


# --- results document --------------------------------------------------------------------------------------------


def _f(x: float | None, digits: int = 3) -> str:
    return "—" if x is None else f"{x:.{digits}f}"


def _version(dist: str) -> str:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return "?"


def _tesseract_version() -> str:
    import pytesseract

    try:
        return str(pytesseract.get_tesseract_version())
    except (OSError, RuntimeError, ValueError):  # metadata only: the benchmark itself would have failed earlier
        return "?"


_RAPID_MODELS = ("ch_PP-OCRv5_det_mobile.onnx", "eslav_PP-OCRv5_rec_mobile.onnx",
                 "ch_ppocr_mobile_v2.0_cls_mobile.onnx")


def _model_weights() -> str:
    """Approximate size of the recognition models (spec 7.2, criterion 3: installation weight); metadata only."""
    import pytesseract

    parts = []
    spec = importlib.util.find_spec("rapidocr")
    if spec and spec.submodule_search_locations:
        models = Path(next(iter(spec.submodule_search_locations))) / "models"
        size = sum((models / n).stat().st_size for n in _RAPID_MODELS if (models / n).exists())
        parts.append(f"RapidOCR — модели {size / 2**20:.1f} МБ + onnxruntime")
    try:
        listing = subprocess.run([pytesseract.pytesseract.tesseract_cmd, "--list-langs"], capture_output=True,
                                 text=True, timeout=30, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        listing = ""
    found = re.search(r'"(.+?)"', listing)
    rus = Path(found.group(1)) / "rus.traineddata" if found else None
    if rus and rus.exists():
        parts.append(f"Tesseract — `rus.traineddata` {rus.stat().st_size / 2**20:.1f} МБ + бинарник (системный пакет)")
    return "; ".join(parts) or "—"


def _load() -> str:
    try:
        return " / ".join(f"{x:.1f}" for x in os.getloadavg())
    except OSError:
        return "?"


def run_metadata(scans: Sequence[SyntheticScan], n_docs: int, seed: int, real_pages: int,
                 seconds_total: float, load_at_start: str = "?") -> dict[str, str]:
    rotated = sum(1 for s in scans if s.meta.get("rot90"))
    return {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "machine": (f"{platform.system()} {platform.release()} {platform.machine()}, {os.cpu_count()} CPU, "
                    f"Python {platform.python_version()}; load average 1/5/15 мин в начале прогона: {load_at_start}"),
        "tesseract": f"Tesseract {_tesseract_version()} (`rus`, `--psm 6`), pytesseract {_version('pytesseract')}",
        "rapidocr": (f"RapidOCR {_version('rapidocr')} (PP-OCRv5 mobile: det + eslav rec), "
                     f"onnxruntime {_version('onnxruntime')}"),
        "libs": f"OpenCV {cv2.__version__}, PyMuPDF {_version('pymupdf')}, NumPy {np.__version__}",
        "weights": _model_weights(),
        "pages": str(len(scans)),
        "docs": str(n_docs),
        "seed": str(seed),
        "rotated": str(rotated),
        "real_pages": str(real_pages),
        "duration": f"{seconds_total / 60:.1f}",
    }


_CURVE_TS = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99)


def render_markdown(rows: Sequence[BenchRow], winner: BenchRow, meta: dict[str, str]) -> str:
    rec = winner.recommended_threshold
    first = rows[0] if rows else winner
    lines = [
        "# E1 — выбор OCR-движка по замеру",
        "",
        (f"Прогон: {meta['date']}, длительность {meta['duration']} мин. Команда: "
         "`uv run python -m contract_analyst.eval.ocr_bench --data data/raw --out docs/experiments/e1_ocr.md`."),
        "Документ содержит только метрики: ни текстов, ни номеров, ни имён файлов (документы конфиденциальны).",
        "",
        "## Варианты",
        "",
        ("Tesseract 5 (`rus`) и PaddleOCR PP-OCRv5 eslav через RapidOCR/ONNX, каждый без предобработки (`none`) и "
         "с выравниванием и бинаризацией Sauvola (`deskew_sauvola`). Не сравнивались (спека §7.2): EasyOCR — тянет "
         "PyTorch и медленный на CPU; VLM-читатель — не используется. Облачный OCR исключён: документы не покидают "
         "машину."),
        "",
        "## Правило выбора (записано до замера: спека §7.2, план Task 18)",
        "",
        ("1. Максимум покрытия критичных токенов при пороге уверенности, дающем точность принятых ≥ 0.99 "
         "(`crit_coverage_at_p99`)."),
        f"2. При ничьей (разница ≤ {TIE}) — меньший CER.",
        "3. Затем — меньшее время на страницу.",
        "",
        ("Критичный токен — слово (без знаков препинания по краям), целиком совпадающее с номером документа "
         "(`D\\d{9}(-\\d{2})?`) или числом (`\\d[\\d.,]*\\d|\\d`): даты, суммы, номера пунктов. Искажённое слово "
         "(например, номер с кириллической «Д») токеном не считается: разборщики не примут его за значение, "
         "поэтому оно снижает покрытие, а не точность. "
         "Токен OCR принят, если уверенность его слова ≥ t (у RapidOCR уверенность — по строке). "
         "Точность(t) — доля принятых токенов, которые есть среди критичных токенов эталона той же страницы "
         "(сопоставление мультимножеств: каждый токен эталона засчитывается не больше одного раза). "
         "Покрытие(t) — верно принятые токены / все критичные токены эталона. Обе величины считаются "
         "по всем страницам сразу. "
         f"Порог t — наименьший на сетке 0.50…0.99 (шаг 0.01), при котором точность ≥ {TARGET_PRECISION}; "
         "если такого нет, покрытие = 0. Порог в конфиг = min(t + 0.05, 0.99) — запас по §6.14."),
        "",
        f"## Результаты: синтетические сканы ({meta['pages']} стр.)",
        "",
        ("| Движок | Предобработка | CER | WER | Покрытие при точности ≥ 0.99 | Порог t | Точность при t "
         "| Порог в конфиг | Покрытие / точность при пороге конфига | с/стр |"),
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        mark = " **(победитель)**" if r is winner else ""
        lines.append(
            f"| `{r.engine}`{mark} | `{r.mode}` | {r.cer:.4f} | {r.wer:.4f} | {r.crit_coverage_at_p99:.3f} "
            f"| {_f(r.threshold, 2)} | {_f(r.crit_precision)} | {_f(r.recommended_threshold, 2)} "
            f"| {_f(r.rec_coverage)} / {_f(r.rec_precision)} | {r.sec_per_page:.2f} |")
    lines += [
        "",
        f"Критичных токенов в эталоне: {first.n_crit_truth}, из них номеров документов: {first.n_doc_numbers}.",
        "",
        f"**Победитель: `{winner.engine}` + `{winner.mode}`.** {_why(rows, winner)}",
        "",
        "Значения для `config/default.yaml`:",
        "",
        "```yaml",
        f"ocr_engine: {winner.engine}",
        f"ocr_preprocess: {winner.mode}",
        f"ocr_word_conf_threshold: {_f(rec, 2) if rec is not None else '<не откалиброван>'}",
        "```",
        "",
    ]
    if rec is None:
        lines += [("Ни один порог на сетке не дал точности ≥ 0.99: порог из конфига не меняется, "
                   "значения из сканов подтверждаются только другими документами."), ""]
    notes = _observations(rows)
    if notes:
        lines += ["## Наблюдения", "", *[f"- {note}" for note in notes], ""]
    lines += [
        "## Когда пересмотреть выбор",
        "",
        ("- Появились расшифровки реальных сканов (Task 17): та же команда добавит их метрики. Если на реальных "
         "страницах варианты расположатся иначе, выбор обсуждается заново: правило для такого расхождения заранее "
         "не записано."),
        ("- Исправлены автоповорот или порядок строк, добавлена нормализация номеров документов после OCR: замер "
         "повторяется с тем же seed (те же страницы и деградации), порог пересчитывается."),
        "- Другая машина или ОС: время на страницу нужно перемерить. На выбор оно влияет только при ничьей по правилам 1–2.",
        "",
    ]
    lines += [
        "## Диагностика",
        "",
        ("| Вариант | WER | Слова эталона, которых нет в OCR (без учёта порядка) "
         "| Повёрнутые на 90°: поворот исправлен | CER на них "
         "| Номера документов прочитаны точно (при любой уверенности) |"),
        "|---|---|---|---|---|---|",
    ]
    lines += [(f"| `{r.engine}` + `{r.mode}` | {r.wer:.4f} | {_f(r.wer_unordered, 4)} | {_f(r.rotation_fixed, 2)} "
               f"| {_f(r.cer_rotated, 4)} | {_f(r.doc_number_recall)} |") for r in rows]
    lines += [
        "",
        "## Кривая: точность / покрытие принятых критичных токенов",
        "",
        "| Вариант | " + " | ".join(f"t={t:.2f}" for t in _CURVE_TS) + " |",
        "|---|" + "---|" * len(_CURVE_TS),
    ]
    for r in rows:
        by_t = {p.t: p for p in r.curve}
        cells = [f"{_f(by_t[t].precision)} / {_f(by_t[t].coverage)}" if t in by_t else "—" for t in _CURVE_TS]
        lines.append(f"| `{r.engine}` + `{r.mode}` | " + " | ".join(cells) + " |")
    lines += ["", "## Реальные сканы (расшифровки из эталона)", ""]
    if any(r.n_real_pages for r in rows):
        lines += [
            f"Страниц: {first.n_real_pages}. Порог — рекомендованный для варианта (t + 0.05).",
            "",
            "| Движок | Предобработка | CER | WER | Точность / покрытие крит. токенов при пороге конфига |",
            "|---|---|---|---|---|",
        ]
        lines += [(f"| `{r.engine}` | `{r.mode}` | {_f(r.real_cer, 4)} | {_f(r.real_wer, 4)} "
                   f"| {_f(r.real_crit_precision)} / {_f(r.real_crit_coverage)} |") for r in rows]
    else:
        lines.append("Ожидается: в `data/golden/golden.yaml` ещё нет расшифровок сканов (`scans`, Task 17). "
                     "Эту часть добавит основная сессия: тот же запуск, когда расшифровки появятся, "
                     "дополнит документ таблицей CER/WER на реальных страницах.")
    lines += [
        "",
        "## Данные и условия прогона",
        "",
        (f"- Синтетические сканы: {meta['pages']} страниц из {meta['docs']} PDF с текстовым слоем; "
         f"seed = {meta['seed']}. Страницы выбраны случайно, по кругу между документами. Страницы, где картинки "
         "занимают больше 1% площади, исключены: их содержимого нет в текстовом слое. Мелкие картинки (логотипы) "
         "закрашены белым."),
        (f"- Рендер {SYNTH_DPI} dpi. Деградации: поворот на 90° у {meta['rotated']} из {meta['pages']} страниц "
         f"(25%), перекос ±{SKEW_DEG:g}°, размытие σ ∈ [{BLUR_SIGMA[0]:g}, {BLUR_SIGMA[1]:g}], "
         f"шум σ ∈ [{NOISE_SIGMA[0]:g}, {NOISE_SIGMA[1]:g}], JPEG с качеством {JPEG_QUALITY[0]}–{JPEG_QUALITY[1]}, "
         f"масштаб {SCALE[0]:g}–{SCALE[1]:g}. Уменьшенная страница возвращается к размеру {SYNTH_DPI} dpi: "
         "так читатель растрирует скан низкого разрешения при фиксированном dpi."),
        ("- Эталон — текстовый слой страницы после `normalize_text`; текст OCR нормализуется так же. "
         "CER и WER — `eval.text_metrics` (переводы строк не считаются), по всем страницам с весом длины эталона."),
        ("- Распознавание — `ocr_page`, как в читателе: с автоповоротом и выбранной предобработкой. "
         "Время — среднее на страницу после прогрева модели. Варианты чередуются постранично: каждая страница "
         "проходит все варианты, и первый вариант каждый раз сдвигается. Поэтому изменение загрузки машины "
         f"влияет на все варианты одинаково. Средняя загрузка (load average за 1 мин) во время замера: {_loads(rows)}."),
        f"- Реальных страниц-сканов с расшифровкой: {meta['real_pages']}.",
        f"- Машина: {meta['machine']}.",
        f"- Движки: {meta['tesseract']}; {meta['rapidocr']}.",
        f"- Библиотеки: {meta['libs']}.",
        f"- Вес моделей (справочно, критерий 3 спеки): {meta['weights']}.",
        "",
    ]
    return "\n".join(lines)


def _loads(rows: Sequence[BenchRow]) -> str:
    loads = [r.load_avg for r in rows if r.load_avg is not None]
    if not loads:
        return "—"
    low, high = min(loads), max(loads)
    return f"{low:.1f}" if round(low, 1) == round(high, 1) else f"{low:.1f}–{high:.1f} по вариантам"


def _observations(rows: Sequence[BenchRow]) -> list[str]:
    """Data-driven remarks for the reader of the results; each one states the numbers it rests on."""
    notes = []
    recalls = [r.doc_number_recall for r in rows if r.doc_number_recall is not None]
    if recalls and max(recalls) < 0.5:
        notes.append(
            f"Номера документов `D…` со сканов плохо читают все варианты: точно прочитано не больше {max(recalls):.0%} "
            "номеров даже без порога уверенности. Искажённое слово не похоже на номер, поэтому разборщики его не "
            "примут: номер со скана будет `absent`, а не неверным. Это задача нормализации после OCR, а не выбора "
            "движка.")
    for r in rows:
        name = f"`{r.engine}` + `{r.mode}`"
        defined = [p for p in r.curve if p.precision is not None]
        if r.threshold is None and defined:
            best = max(defined, key=lambda p: p.precision)
            notes.append(f"{name}: точность {TARGET_PRECISION} не достигается ни при каком пороге; лучшая — "
                         f"{best.precision:.3f} при t = {best.t:.2f}.")
        elif r.rec_coverage is not None and r.rec_coverage == 0:
            notes.append(
                f"{name}: при пороге в конфиг {_f(r.recommended_threshold, 2)} не принимается ни один критичный токен "
                f"(при t = {_f(r.threshold, 2)} покрытие {r.crit_coverage_at_p99:.3f}): уверенность слов этого "
                "движка до порога с запасом не доходит.")
        elif r.rec_coverage is not None and r.rec_coverage < r.crit_coverage_at_p99 - 0.1:
            notes.append(f"{name}: запас +0.05 снижает покрытие с {r.crit_coverage_at_p99:.3f} до {r.rec_coverage:.3f}.")
    for r in rows:
        if r.wer_unordered is not None and r.wer - r.wer_unordered > 0.1:
            notes.append(
                f"`{r.engine}` + `{r.mode}`: WER {r.wer:.3f}, а без учёта порядка слов — {r.wer_unordered:.3f}. "
                "Большая часть ошибок — порядок, в котором склеены строки, а не распознавание слов; он портит CER "
                "и мешает шлюзу найти цитату, которая переходит через строку.")
    for r in rows:
        if r.rotation_fixed is not None and r.rotation_fixed < 1 and r.cer_rotated is not None:
            notes.append(
                f"`{r.engine}` + `{r.mode}`: `ocr_page` вернул в нормальное положение {r.rotation_fixed:.0%} страниц, "
                f"повёрнутых на 90°; CER на этих страницах {r.cer_rotated:.3f} при среднем {r.cer:.3f}.")
    return notes


def _why(rows: Sequence[BenchRow], winner: BenchRow) -> str:
    best = max(r.crit_coverage_at_p99 for r in rows)
    tied = [r for r in rows if best - r.crit_coverage_at_p99 <= TIE + _EPS]
    if len(tied) == 1:
        return f"Правило 1: наибольшее покрытие критичных токенов ({winner.crit_coverage_at_p99:.3f})."
    same_cer = [r for r in tied if r.cer == winner.cer]
    if len(same_cer) > 1:
        return f"Правила 1–2 — ничья ({len(tied)} варианта), правило 3: быстрее ({winner.sec_per_page:.2f} с/стр)."
    return (f"Правило 1 — ничья между {len(tied)} вариантами (разница покрытия ≤ {TIE}), "
            f"правило 2: меньший CER ({winner.cer:.4f}).")


# --- entry point -------------------------------------------------------------------------------------------------


def _input_pdfs(data_dir: Path) -> list[Path]:
    from contract_analyst.ingest.inventory import list_inputs
    from contract_analyst.models import DocFormat

    return [Path(f.path) for f in list_inputs(data_dir) if f.format == DocFormat.PDF and f.duplicate_of is None]


def run_benchmark(data_dir: Path, out: Path, golden: Path | None = None, n_pages: int = 40, seed: int = 0,
                  engines: Sequence[str] = ENGINES, modes: Sequence[str] = MODES,
                  log: Log | None = None) -> tuple[list[BenchRow], BenchRow]:
    """Build the synthetic set, score every variant, write the results document; returns rows and winner."""
    from contract_analyst.config import load_settings

    started, load_at_start = time.perf_counter(), _load()
    pdfs = _input_pdfs(data_dir)
    scans = make_synthetic_scans(pdfs, n_pages=n_pages, seed=seed)
    golden = golden if golden is not None else data_dir.parent / "golden" / "golden.yaml"
    real = load_real_pages(golden, data_dir, dpi=load_settings().ocr_dpi)
    n_docs = len({s.meta["doc_index"] for s in scans})
    if log:
        log(f"synthetic pages: {len(scans)} from {n_docs} documents; real scanned pages: {len(real)}")
    rows = bench(engines, modes, scans, real, log=log)
    winner = select_winner(rows)
    meta = run_metadata(scans, n_docs, seed, len(real), time.perf_counter() - started, load_at_start)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_markdown(rows, winner, meta), encoding="utf-8")
    return rows, winner


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m contract_analyst.eval.ocr_bench",
                                     description="E1: OCR engine x preprocessing benchmark (metrics only)")
    parser.add_argument("--data", type=Path, required=True, help="input folder (data/raw)")
    parser.add_argument("--out", type=Path, default=Path("docs/experiments/e1_ocr.md"))
    parser.add_argument("--golden", type=Path, default=None,
                        help="golden.yaml with hand-typed scans (default: <data>/../golden/golden.yaml)")
    parser.add_argument("--pages", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--engines", default=",".join(ENGINES))
    parser.add_argument("--modes", default=",".join(MODES))
    args = parser.parse_args(argv)

    def log(message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    rows, winner = run_benchmark(args.data, args.out, golden=args.golden, n_pages=args.pages, seed=args.seed,
                                 engines=[e for e in args.engines.split(",") if e],
                                 modes=[m for m in args.modes.split(",") if m], log=log)
    for r in rows:
        print(f"{r.engine:10s} {r.mode:15s} {_row_summary(r)}")
    print(f"winner: {winner.engine} + {winner.mode}; config threshold {_f(winner.recommended_threshold, 2)}")
    print(f"written {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
