"""Golden set tooling: prefill from deterministic parsing, text dumps for annotators."""

from __future__ import annotations

from pathlib import Path

import cv2

from contract_analyst.eval.golden import GoldenDocument, GoldenSet, GoldenValue
from contract_analyst.models import DocKind, DocumentText, PageMethod
from contract_analyst.ocr.render import render_page
from contract_analyst.parsing.header import detect_draft, parse_header

_KIND = {"contract": DocKind.MAIN, "supplement": DocKind.SUPPLEMENT, "agreement": DocKind.FRAMEWORK,
         "unknown": DocKind.UNKNOWN}


def _rel(doc: DocumentText, root: Path | None) -> str:
    if root is None:
        return doc.file.name
    try:
        return str(Path(doc.file.path).relative_to(root))
    except ValueError:
        return doc.file.name


def _gv(value, quote: str | None) -> GoldenValue:
    if value is None:
        return GoldenValue(status="absent")
    text = value.isoformat() if hasattr(value, "isoformat") else str(value)
    return GoldenValue(value=text, page=1, quote=quote)


def prefill(docs: list[DocumentText], root: Path | None = None) -> GoldenSet:
    documents = []
    for doc in docs:
        first = doc.pages[0].text if doc.pages else ""
        h = parse_header(first)
        documents.append(GoldenDocument(
            file=_rel(doc, root), doc_id=doc.doc_id, kind=_KIND[h.title_kind], is_draft=detect_draft(doc.file.name, first),
            number=_gv(h.number, h.spans.get("number")), doc_date=_gv(h.doc_date, h.spans.get("doc_date")),
            city=_gv(h.city, h.spans.get("city")), parent_number=_gv(h.parent_number, h.spans.get("parent")),
            parent_date=_gv(h.parent_date, h.spans.get("parent"))))
    return GoldenSet(documents=documents)


def dump_texts(docs: list[DocumentText], out_dir: Path, png_dpi: int = 150) -> list[Path]:
    """One text file per document with page markers; OCR pages also as PNG for visual checks."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for doc in docs:
        lines = [f"# file: {doc.file.name}", f"# doc_id: {doc.doc_id}", f"# status: {doc.status}"]
        for p in doc.pages:
            conf = f", ocr_conf={p.quality.ocr_mean_conf:.2f}" if p.quality.ocr_mean_conf is not None else ""
            lines.append(f"\n=== PAGE {p.number} ({p.method.value}{conf}) ===\n{p.text}")
            if p.method == PageMethod.OCR and Path(doc.file.path).suffix.lower() == ".pdf":
                png = out_dir / "png" / f"{doc.doc_id}_p{p.number}.png"
                png.parent.mkdir(exist_ok=True)
                image = render_page(Path(doc.file.path), p.number, png_dpi)
                cv2.imwrite(str(png), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        path = out_dir / f"{doc.doc_id}.txt"
        path.write_text("\n".join(lines), encoding="utf-8")
        written.append(path)
    return written
