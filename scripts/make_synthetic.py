"""Build the held-out synthetic set: synthetic/sources/**/*.md -> synthetic/build/**.

All parties and document numbers are fictional (D99xxxxxxx-01) and the wording is our own. Source format: one
paragraph per line, wrapped here to the page width; a `<!-- page -->` line starts a new page, other `<!-- ... -->`
lines are comments. Files are made by tests/fixtures/builders.py: make_text_pdf, make_scan_pdf (seed 0, moderate
noise) and make_docx.

doc_id is sha256[:12] of a file, so the output is byte-reproducible: PDF trailer IDs and DOCX zip timestamps are
fixed and embedded fonts are subset. Scan pages are then stored as JPEG, the way a scanner stores them (the builder
keeps the noisy page lossless, about 4 MB a page). When a rebuild changes a file's doc_id (an edited source, other
library versions), the old id is replaced with the new one in synthetic/golden.yaml.

Usage: uv run python scripts/make_synthetic.py
"""

from __future__ import annotations

import hashlib
import io
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import pymupdf
import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "synthetic" / "sources"
BUILD = ROOT / "synthetic" / "build"
GOLDEN = ROOT / "synthetic" / "golden.yaml"

SCAN_NOISE = 12.0  # top of the E1 noise range (sigma 0..12)
SCAN_SEED = 0
SCAN_JPEG_QUALITY = 40  # bottom of the E1 JPEG quality range (40..90)
FONT_SIZE = 12  # the builders write 12 pt FiraGO ("figo")
MAX_LINE_WIDTH = 515.0  # pt: a 595 pt page minus the builders' 40 pt left margin and the same on the right
MAX_LINES = 42  # the builders put lines 18 pt apart from y = 50 on an 842 pt page
PAGE_BREAK = "<!-- page -->"
ZIP_TIME = (1980, 1, 1, 0, 0, 0)


@dataclass(frozen=True)
class Item:
    source: str  # relative to SOURCES
    output: str  # relative to BUILD; the golden `file` field uses the same path
    kind: Literal["text_pdf", "scan_pdf", "docx"]


# File names are deliberately unhelpful (T3-style): ДС1/ДС2 contradict the real order, the main dealer contract is
# named by its number without the -01 suffix. Spaces and "+" in names exercise path handling.
MANIFEST = (
    Item("lease/contract_D990100001-01.md", "lease/Договор аренды (скан).pdf", "scan_pdf"),
    Item("lease/supplement_D990100002-01.md", "lease/ДС 1 аренда.pdf", "text_pdf"),
    Item("lease/supplement_D990100003-01.md", "lease/ДС 2 срок + пролонгация.pdf", "text_pdf"),
    Item("services/contract_D990200001-01.md", "services/Договор оказания услуг.docx", "docx"),
    Item("services/supplement_D990200002-01.md", "services/Расторжение.pdf", "text_pdf"),
    Item("dealer/contract_D990300001-01.md", "dealer/D990300001.pdf", "text_pdf"),
    Item("dealer/supplement_D990300002-01.md", "dealer/ДС2_финал.pdf", "text_pdf"),
    Item("dealer/supplement_D990300003-01.md", "dealer/ДС1.pdf", "text_pdf"),
    Item("dealer/supplement_D990300004-01.md", "dealer/ДС3_консервация.pdf", "text_pdf"),
)


def _trim(lines: list[str]) -> list[str]:
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def read_source(path: Path) -> list[list[str]]:
    """Paragraphs per page: comments dropped, blank lines kept inside a page."""
    pages: list[list[str]] = [[]]
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.strip() == PAGE_BREAK:
            pages.append([])
        elif not raw.strip().startswith("<!--"):
            pages[-1].append(raw.rstrip())
    return [_trim(page) for page in pages]


def wrap(paragraph: str, font: pymupdf.Font) -> list[str]:
    """Greedy wrap by rendered width. Lines break only at spaces, so hyphenated words are never split."""
    if not paragraph.strip():
        return [""]
    words = paragraph.split(" ")
    lines, line = [], words[0]
    for word in words[1:]:
        candidate = f"{line} {word}"
        if font.text_length(candidate, fontsize=FONT_SIZE) <= MAX_LINE_WIDTH:
            line = candidate
        else:
            lines.append(line.rstrip())
            line = word
    lines.append(line.rstrip())
    return lines


def layout(source: Path) -> list[str]:
    """Page texts for the PDF builders."""
    font = pymupdf.Font("figo")
    pages = []
    for number, paragraphs in enumerate(read_source(source), start=1):
        lines = [line for paragraph in paragraphs for line in wrap(paragraph, font)]
        if len(lines) > MAX_LINES:
            raise SystemExit(f"{source.relative_to(ROOT)}: page {number} has {len(lines)} lines, max {MAX_LINES}")
        pages.append("\n".join(lines))
    return pages


def docx_paragraphs(source: Path) -> list[str]:
    pages = read_source(source)
    if len(pages) != 1:
        raise SystemExit(f"{source.relative_to(ROOT)}: a DOCX source has no pages, remove {PAGE_BREAK}")
    return [p for p in pages[0] if p.strip()]


def finalize_pdf(path: Path, source_text: str, scan: bool) -> None:
    """Make the PDF reproducible: JPEG scan pages or subset fonts, then a content-derived trailer ID."""
    doc = pymupdf.open(path)
    if scan:
        for page in doc:
            for xref, *_ in page.get_images():
                pix = pymupdf.Pixmap(doc, xref)
                rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
                bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
                ok, jpeg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, SCAN_JPEG_QUALITY])
                if not ok:
                    raise SystemExit(f"{path}: JPEG encoding failed")
                page.replace_image(xref, stream=jpeg.tobytes())
    else:
        doc.subset_fonts()
    digest = hashlib.md5(source_text.encode("utf-8")).hexdigest().upper()
    doc.xref_set_key(-1, "ID", f"[<{digest}><{digest}>]")
    data = doc.tobytes(garbage=3, deflate=True, no_new_id=True)
    doc.close()
    path.write_bytes(data)


def finalize_docx(path: Path) -> None:
    """Rewrite the zip with fixed timestamps: python-docx stamps every entry with the current time."""
    with zipfile.ZipFile(path) as src:
        entries = [(info.filename, src.read(info)) for info in src.infolist()]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as dst:
        for name, data in entries:
            info = zipfile.ZipInfo(name, date_time=ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o644 << 16
            dst.writestr(info, data)
    path.write_bytes(buffer.getvalue())


def sync_golden(ids: dict[str, str]) -> None:
    """Replace stale doc_ids in golden.yaml; documents are matched by their `file` path."""
    if not GOLDEN.exists():
        return
    text = GOLDEN.read_text(encoding="utf-8")
    updated = text
    for doc in (yaml.safe_load(text) or {}).get("documents", []):
        new = ids.get(doc.get("file"))
        if new is None:
            print(f"warning: golden.yaml lists {doc.get('file')}, which the manifest does not build")
        elif doc["doc_id"] != new:
            updated = re.sub(rf"\b{re.escape(doc['doc_id'])}\b", new, updated)
            print(f"golden.yaml: {doc['file']}: doc_id {doc['doc_id']} -> {new}")
    if updated != text:
        GOLDEN.write_text(updated, encoding="utf-8")


def main() -> None:
    sys.path.insert(0, str(ROOT))
    from tests.fixtures.builders import make_docx, make_scan_pdf, make_text_pdf

    ids: dict[str, str] = {}
    for item in MANIFEST:
        source, out = SOURCES / item.source, BUILD / item.output
        out.parent.mkdir(parents=True, exist_ok=True)
        if item.kind == "docx":
            make_docx(out, docx_paragraphs(source))
            finalize_docx(out)
        else:
            pages = layout(source)
            if item.kind == "scan_pdf":
                make_scan_pdf(out, pages, noise=SCAN_NOISE, seed=SCAN_SEED)
            else:
                make_text_pdf(out, pages)
            finalize_pdf(out, "\f".join(pages), scan=item.kind == "scan_pdf")
        ids[item.output] = hashlib.sha256(out.read_bytes()).hexdigest()[:12]
        print(f"{ids[item.output]}  {item.kind:<8}  {out.stat().st_size // 1024:>4} KB  {item.output}")
    built = {BUILD / item.output for item in MANIFEST}
    for extra in sorted(p for p in BUILD.rglob("*") if p.is_file() and p not in built and not p.name.startswith(".")):
        print(f"warning: {extra.relative_to(ROOT)} is not in the manifest; the synthetic test will read it too")
    sync_golden(ids)


if __name__ == "__main__":
    main()
