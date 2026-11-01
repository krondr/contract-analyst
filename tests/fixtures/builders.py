"""Builders for synthetic test documents. All numbers and parties are fictional (D99xxxxxxx)."""

from __future__ import annotations

from pathlib import Path

import cv2
import docx
import numpy as np
import pymupdf

from contract_analyst.models import OcrWord
from contract_analyst.ocr.base import OcrResult

FONT = "figo"  # FiraGO from pymupdf-fonts: covers Cyrillic
_LINE_HEIGHT = 18


def _write_lines(page: pymupdf.Page, text: str) -> None:
    for i, line in enumerate(text.split("\n")):
        page.insert_text((40, 50 + i * _LINE_HEIGHT), line, fontname=FONT, fontsize=12)


def _page_height(text: str) -> int:
    return max(200, 80 + _LINE_HEIGHT * (text.count("\n") + 1))


def render_text_image(text: str, dpi: int = 200) -> np.ndarray:
    """Render text as an RGB uint8 image, the way a clean scan would look."""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=_page_height(text))
    _write_lines(page, text)
    pix = page.get_pixmap(dpi=dpi)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3].copy()


def make_text_pdf(path: Path, pages: list[str]) -> Path:
    doc = pymupdf.open()
    for text in pages:
        _write_lines(doc.new_page(width=595, height=842), text)
    doc.save(path)
    return path


def _degrade(image: np.ndarray, rotate: int, noise: float, seed: int) -> np.ndarray:
    out = np.rot90(image, k=rotate // 90).copy() if rotate else image.copy()
    if noise:
        rng = np.random.default_rng(seed)
        out = np.clip(out.astype(np.int16) + rng.normal(0, noise, out.shape), 0, 255).astype(np.uint8)
    return out


def _insert_image_page(doc: pymupdf.Document, image: np.ndarray) -> None:
    ok, png = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    assert ok
    h, w = image.shape[:2]
    page = doc.new_page(width=w * 72 / 150, height=h * 72 / 150)
    page.insert_image(page.rect, stream=png.tobytes())


def make_scan_pdf(path: Path, pages: list[str], rotate: int = 0, noise: float = 0.0, seed: int = 0) -> Path:
    doc = pymupdf.open()
    for text in pages:
        _insert_image_page(doc, _degrade(render_text_image(text, dpi=150), rotate, noise, seed))
    doc.save(path)
    return path


def make_docx(path: Path, paragraphs: list[str]) -> Path:
    document = docx.Document()
    for p in paragraphs:
        document.add_paragraph(p)
    document.save(path)
    return path


def build_mixed_pdf(path: Path) -> Path:
    """Page 1: text layer; page 2: scanned image only; page 3: blank."""
    doc = pymupdf.open()
    _write_lines(doc.new_page(width=595, height=842),
                 "Договор поставки № D990000001-01\n1. Предмет договора: поставка товара по заявкам покупателя.")
    _insert_image_page(doc, render_text_image("2. Цена товара составляет 1500 рублей.", dpi=150))
    doc.new_page(width=595, height=842)
    doc.save(path)
    return path


class FakeOcrEngine:
    name = "fake"

    def __init__(self, text: str = "распознанный текст 1500", conf: float = 0.95):
        self.text, self.conf = text, conf

    def recognize(self, image: np.ndarray) -> OcrResult:
        words = [OcrWord(text=w, conf=self.conf) for w in self.text.split()]
        return OcrResult(text=self.text, words=words, mean_conf=self.conf, rotation=0)


def text_doc(pages: list[str], doc_id: str = "doc000000001", name: str = "mem.pdf"):
    """In-memory DocumentText made of text-layer pages."""
    from contract_analyst.ingest.normalize import normalize_text
    from contract_analyst.ingest.segment import segment_clauses
    from contract_analyst.models import DocFormat, DocumentText, InputFile, Page, PageQuality

    built = [Page(number=i, method="text", text=normalize_text(t), quality=PageQuality(chars=len(t)))
             for i, t in enumerate(pages, start=1)]
    file = InputFile(path=name, name=name, sha256=doc_id.ljust(64, "0"), format=DocFormat.PDF, size_bytes=0)
    return DocumentText(doc_id=doc_id, file=file, pages=built, clauses=segment_clauses(built), status="ok")


def ocr_doc(text: str, conf: dict[int, float] | None = None, words: list[str] | None = None,
            doc_id: str = "scan00000001"):
    """In-memory DocumentText with one OCR page. Its words are the text's tokens (or `words`), each with
    confidence 0.97 unless `conf` maps the word's index to another value."""
    from contract_analyst.models import DocFormat, DocumentText, InputFile, Page, PageQuality

    tokens = text.split() if words is None else words
    page = Page(number=1, method="ocr", text=text, words=[OcrWord(text=w, conf=(conf or {}).get(i, 0.97))
                                                           for i, w in enumerate(tokens)],
                quality=PageQuality(chars=len(text), ocr_mean_conf=0.97, low_conf_ratio=0.0))
    file = InputFile(path="scan.pdf", name="scan.pdf", sha256=doc_id.ljust(64, "0"), format=DocFormat.PDF,
                     size_bytes=0)
    return DocumentText(doc_id=doc_id, file=file, pages=[page], clauses=[], status="ok")


def acc(value, doc_id: str = "d", quote: str = "цитата", page: int = 1):
    """Accepted Extracted value with a source."""
    from contract_analyst.models import Extracted, Source, ValueStatus

    return Extracted(value=value, status=ValueStatus.ACCEPTED, source=Source(doc_id=doc_id, page=page, quote=quote))


def make_record(doc_id: str, number: str | None, kind: str = "supplement", doc_date=None, parent: str | None = None,
                parent_date=None, effective=None, effective_to=None, ops=(), terms: dict | None = None, rates=(), party_names=(),
                category: str = "other", is_draft: bool = False, file_name: str | None = None, rejected_date=None):
    """DocumentRecord for assembly tests. ops: (OpType, target, effective_from); rates: (label, amount, from, to)."""
    from decimal import Decimal

    from contract_analyst.models import (
        AmendmentOp,
        DocumentRecord,
        Extracted,
        ParentRef,
        Party,
        RateRow,
        Source,
        ValueStatus,
    )

    def absent():
        return Extracted(status=ValueStatus.ABSENT)

    src = Source(doc_id=doc_id, page=1, quote="цитата")
    date_ext = acc(doc_date, doc_id) if doc_date else absent()
    if rejected_date:
        date_ext = Extracted(value=rejected_date, status=ValueStatus.REJECTED, source=src, reason="ocr_low_confidence")
    return DocumentRecord(
        doc_id=doc_id, file_name=file_name or f"{doc_id}.pdf", title=number or doc_id, is_draft=is_draft,
        kind=acc(kind, doc_id), number=acc(number, doc_id) if number else absent(), doc_date=date_ext,
        city=absent(),
        parent=acc(ParentRef(number=parent, date_stated=parent_date), doc_id) if parent else absent(),
        effective_from=acc(effective, doc_id) if effective else absent(),
        effective_to=acc(effective_to, doc_id) if effective_to else absent(), subject_category=acc(category, doc_id),
        parties=[Party(role=f"p{i}", org_type=acc("ООО", doc_id), name=acc(n, doc_id)) for i, n in enumerate(party_names)],
        terms={k: acc(v, doc_id) for k, v in (terms or {}).items()},
        rates=[RateRow(label=label, amount=Decimal(str(amount)), valid_from=vf, valid_to=vt, status=ValueStatus.ACCEPTED,
                       source=src) for label, amount, vf, vt in rates],
        ops=[AmendmentOp(op=op, target=target, effective_from=eff, status=ValueStatus.ACCEPTED, source=src)
             for op, target, eff in ops])
