"""Build the third clean evaluation set: clean3/sources/**/*.md -> clean3/build/**.

A fifth set of fictional documents (numbers D95xxxxxxx-NN, invented parties, wording not reused from clean/ or
clean2/), written without looking at the system's code, rules, prompts or the other sets' texts. Only the final
result is measured on it (clean3/README.md).

Same source format and builders as scripts/make_clean.py and scripts/make_clean2.py: one paragraph per line, a
`<!-- page -->` line starts a new page, other `<!-- ... -->` lines are comments; files are made by
tests/fixtures/builders.py (make_text_pdf, make_scan_pdf with noise, make_docx) and finalized for byte-reproducibility,
so doc_id (sha256[:12] of a file) is stable. When a rebuild changes a file's doc_id, the old id is replaced with the
new one in clean3/golden.yaml.

Usage: uv run python scripts/make_clean3.py
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "clean3" / "sources"
BUILD = ROOT / "clean3" / "build"
GOLDEN = ROOT / "clean3" / "golden.yaml"

SCAN_NOISE = 12.0  # top of the E1 noise range, as in the other sets
SCAN_SEED = 17  # synthetic: seed 0, held-out: 7, clean: 11, clean2: 13


def manifest(item_cls):
    # File names are partly unhelpful on purpose: nothing may be inferred from them.
    return (
        item_cls("lab/contract_D951000001-01.md", "lab/Рамочный договор испытания.pdf", "text_pdf"),
        item_cls("lab/supplement_D951000002-01.md", "lab/ДС 2 прейскурант.docx", "docx"),
        item_cls("lab/supplement_D951000003-01.md", "lab/Доп соглашение (продление).pdf", "text_pdf"),
        item_cls("lab/supplement_D951000004-01.md", "lab/Скан_20261021.pdf", "scan_pdf"),
        item_cls("vending/contract_D952000001-01.md", "vending/D952000001-01 аренда мест.docx", "docx"),
        item_cls("vending/supplement_D952000002-01.md", "vending/новая редакция.pdf", "text_pdf"),
        item_cls("vending/supplement_D952000003-01.md", "vending/Документ1.pdf", "text_pdf"),
        item_cls("camp/contract_D953000001-01.md", "camp/агентский договор Кочергина.pdf", "text_pdf"),
        item_cls("camp/supplement_D953000002-01.md", "camp/ДС вознаграждение 2026.docx", "docx"),
        item_cls("camp/termination_D953000003-01.md", "camp/IMG_4471.pdf", "scan_pdf"),
    )


def sync_golden(ids: dict[str, str]) -> None:
    """Replace stale doc_ids in clean3/golden.yaml; documents are matched by their `file` path."""
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
    sys.path.insert(0, str(ROOT / "scripts"))
    from make_synthetic import Item, docx_paragraphs, finalize_docx, finalize_pdf, layout

    from tests.fixtures.builders import make_docx, make_scan_pdf, make_text_pdf

    items = manifest(Item)
    ids: dict[str, str] = {}
    for item in items:
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
    built = {BUILD / item.output for item in items}
    for extra in sorted(
        p for p in BUILD.rglob("*") if p.is_file() and p not in built and not p.name.startswith(".")
    ):
        print(f"warning: {extra.relative_to(ROOT)} is not in the manifest; a run on clean3/build will read it too")
    sync_golden(ids)


if __name__ == "__main__":
    main()
