"""Read any supported input into per-page text: text layer where present, OCR where not."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import cv2
import docx
import pymupdf
from docx.opc.exceptions import PackageNotFoundError

from contract_analyst.config import Settings
from contract_analyst.ingest.normalize import normalize_text
from contract_analyst.ingest.segment import segment_clauses
from contract_analyst.models import DocFormat, DocumentText, InputFile, Page, PageMethod, PageQuality
from contract_analyst.ocr import OcrEngine, ocr_page
from contract_analyst.ocr.render import render_page

READER_VERSION = 2  # bump when read_document returns something else for the same file and settings


def _chars(text: str) -> int:
    return sum(1 for ch in text if not ch.isspace())


def _ocr_to_page(number: int, image, engine: OcrEngine, settings: Settings, mode: str | None = None) -> Page:
    result = ocr_page(image, engine, mode or settings.ocr_preprocess)
    text = normalize_text(result.text)
    low = [w for w in result.words if w.conf < settings.ocr_word_conf_threshold]
    ratio = len(low) / len(result.words) if result.words else 1.0
    quality = PageQuality(chars=_chars(text), ocr_mean_conf=result.mean_conf, low_conf_ratio=ratio,
                          rotation=result.rotation)
    return Page(number=number, method=PageMethod.OCR, text=text, quality=quality, words=result.words)


def _status(pages: list[Page], settings: Settings) -> str:
    if not pages or sum(p.quality.chars for p in pages) == 0:
        return "empty"
    for p in pages:
        if p.method == PageMethod.OCR and (
            (p.quality.ocr_mean_conf or 0.0) < settings.ocr_word_conf_threshold
            or (p.quality.low_conf_ratio or 0.0) > settings.low_quality_ratio
        ):
            return "low_quality"
    return "ok"


def _error(file: InputFile, issue: str) -> DocumentText:
    return DocumentText(doc_id=file.sha256[:12], file=file, pages=[], clauses=[], status="error", issues=[issue])


def _finish(file: InputFile, pages: list[Page], settings: Settings) -> DocumentText:
    return DocumentText(doc_id=file.sha256[:12], file=file, pages=pages, clauses=segment_clauses(pages),
                        status=_status(pages, settings))


def _read_pdf(file: InputFile, engine: OcrEngine, settings: Settings) -> DocumentText:
    try:
        doc = pymupdf.open(file.path)
    except (RuntimeError, ValueError, OSError):  # pymupdf.FileDataError is a RuntimeError
        return _error(file, "corrupted")
    with doc:
        if doc.needs_pass:
            return _error(file, "encrypted")
        if doc.page_count == 0:
            return _error(file, "corrupted")
        pages: list[Page] = []
        for i, page in enumerate(doc, start=1):
            text = normalize_text(page.get_text())
            if _chars(text) >= settings.text_layer_min_chars:
                pages.append(Page(number=i, method=PageMethod.TEXT, text=text, quality=PageQuality(chars=_chars(text))))
            elif page.get_images():
                pages.append(_ocr_to_page(i, render_page(Path(file.path), i, settings.ocr_dpi), engine, settings))
            else:
                pages.append(Page(number=i, method=PageMethod.EMPTY, text=text, quality=PageQuality(chars=_chars(text))))
    return _finish(file, pages, settings)


def _read_docx(file: InputFile, settings: Settings) -> DocumentText:
    try:
        document = docx.Document(file.path)
    except (PackageNotFoundError, zipfile.BadZipFile, KeyError, ValueError, OSError):
        return _error(file, "corrupted")
    parts = [p.text for p in document.paragraphs]
    for table in document.tables:
        parts.extend(" | ".join(cell.text for cell in row.cells) for row in table.rows)
    text = normalize_text("\n".join(parts))
    return _finish(file, [Page(number=1, method=PageMethod.TEXT, text=text, quality=PageQuality(chars=_chars(text)))],
                   settings)


def _frame(path: str, index: int):
    """One frame of an image file as RGB, read alone so a long multi-page TIFF is not held in memory at once."""
    ok, frames = cv2.imreadmulti(path, index, 1, flags=cv2.IMREAD_COLOR)
    return cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB) if ok and frames else None


def _read_image(file: InputFile, engine: OcrEngine, settings: Settings) -> DocumentText:
    pages: list[Page] = []
    for i in range(cv2.imcount(file.path)):  # a multi-page TIFF is a multi-page document; 0 if unreadable
        image = _frame(file.path, i)
        if image is None:
            return _error(file, "corrupted")
        pages.append(_ocr_to_page(i + 1, image, engine, settings))
    return _finish(file, pages, settings) if pages else _error(file, "corrupted")


def _cache_path(cache_dir: Path, file: InputFile, engine: OcrEngine, settings: Settings) -> Path:
    knobs = (READER_VERSION, settings.ocr_preprocess, settings.ocr_dpi, settings.text_layer_min_chars,
             settings.ocr_word_conf_threshold, settings.low_quality_ratio)
    return cache_dir / f"{file.sha256}-{engine.name}-{hashlib.sha256(repr(knobs).encode()).hexdigest()[:12]}.json"


def read_document(file: InputFile, engine: OcrEngine, settings: Settings, cache_dir: Path | None = None) -> DocumentText:
    cached = _cache_path(cache_dir, file, engine, settings) if cache_dir else None
    if cached and cached.exists():
        doc = DocumentText.model_validate_json(cached.read_text(encoding="utf-8"))
        # Segmentation changes more often than OCR does: cut the cached pages again instead of trusting old clauses.
        return doc.model_copy(update={"file": file, "clauses": segment_clauses(doc.pages)})
    if file.format == DocFormat.PDF:
        result = _read_pdf(file, engine, settings)
    elif file.format == DocFormat.DOCX:
        result = _read_docx(file, settings)
    elif file.format == DocFormat.IMAGE:
        result = _read_image(file, engine, settings)
    else:
        result = _error(file, "unsupported_format")
    if cached:
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(result.model_dump_json(), encoding="utf-8")
    return result


def rerecognize_pages(doc: DocumentText, pages: list[int], engine: OcrEngine, mode: str, dpi: int,
                      settings: Settings | None = None) -> DocumentText:
    from contract_analyst.config import load_settings

    settings = settings or load_settings()
    new_pages = list(doc.pages)
    for number in pages:
        if doc.file.format == DocFormat.IMAGE:
            image = _frame(doc.file.path, number - 1)
            if image is None:
                continue
        else:
            image = render_page(Path(doc.file.path), number, dpi)
        new_pages[number - 1] = _ocr_to_page(number, image, engine, settings, mode)
    return doc.model_copy(update={"pages": new_pages, "clauses": segment_clauses(new_pages),
                                  "status": _status(new_pages, settings)})
