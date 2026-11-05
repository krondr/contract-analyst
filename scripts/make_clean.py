"""Build the clean evaluation set: clean/sources/**/*.md -> clean/build/**.

A third set of fictional documents (numbers D97xxxxxxx-NN, invented parties, our own wording), written before the
next round of rule changes and without looking at the system's rules, prompts or the other sets' texts. Only the
result of that round is measured on it (clean/README.md).

Same source format and builders as scripts/make_synthetic.py and scripts/make_holdout.py, whose helpers this script
reuses: one paragraph per line, a `<!-- page -->` line starts a new page, other `<!-- ... -->` lines are comments;
files are made by tests/fixtures/builders.py (make_text_pdf, make_scan_pdf with noise, make_docx) and finalized for
byte-reproducibility, so doc_id (sha256[:12] of a file) is stable. When a rebuild changes a file's doc_id, the old id
is replaced with the new one in clean/golden.yaml.

Usage: uv run python scripts/make_clean.py
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "clean" / "sources"
BUILD = ROOT / "clean" / "build"
GOLDEN = ROOT / "clean" / "golden.yaml"

SCAN_NOISE = 12.0  # top of the E1 noise range, as in the other sets
SCAN_SEED = 11  # the synthetic set uses seed 0, the held-out set seed 7


def manifest(item_cls):
    # File names are partly unhelpful on purpose: nothing may be inferred from them.
    return (
        item_cls("agency/contract_D971000001-01.md", "agency/Агентский договор Ориентир.pdf", "text_pdf"),
        item_cls("agency/supplement_D971000002-01.md", "agency/ДС_2026.pdf", "text_pdf"),
        item_cls("agency/supplement_D971000003-01.md", "agency/доп3.pdf", "text_pdf"),
        item_cls("lease/contract_D972000001-01.md", "lease/Аренда Фабричная 31.docx", "docx"),
        item_cls("lease/supplement_D972000002-01.md", "lease/ДС аренда.pdf", "text_pdf"),
        item_cls("lease/termination_D972000003-01.md", "lease/scan_1002.pdf", "scan_pdf"),
        item_cls("support/contract_D973000001-01.md", "support/Договор ИТ-поддержки D973000001-01.pdf", "text_pdf"),
        item_cls("support/supplement_D973000002-01.md", "support/ДС бэкап.pdf", "text_pdf"),
        item_cls("support/supplement_D973000003-01.md", "support/Документ1.docx", "docx"),
        item_cls("support/supplement_D973000004-01.md", "support/ДС_новое.pdf", "text_pdf"),
    )


def sync_golden(ids: dict[str, str]) -> None:
    """Replace stale doc_ids in clean/golden.yaml; documents are matched by their `file` path."""
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
        print(f"warning: {extra.relative_to(ROOT)} is not in the manifest; a run on clean/build will read it too")
    sync_golden(ids)


if __name__ == "__main__":
    main()
