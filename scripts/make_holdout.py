"""Build the held-out evaluation set: holdout/sources/**/*.md -> holdout/build/**.

A second set of fictional documents (numbers D98xxxxxxx-NN, invented parties, our own wording), written without
looking at the system's rules, prompts or the synthetic set. The system is run on it once, after its rules are frozen.

Same source format and builders as scripts/make_synthetic.py, whose helpers this script reuses: one paragraph per
line, a `<!-- page -->` line starts a new page; files are made by tests/fixtures/builders.py (make_text_pdf,
make_scan_pdf with noise, make_docx) and finalized for byte-reproducibility, so doc_id (sha256[:12] of a file) is
stable. When a rebuild changes a file's doc_id, the old id is replaced with the new one in holdout/golden.yaml.

Usage: uv run python scripts/make_holdout.py
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "holdout" / "sources"
BUILD = ROOT / "holdout" / "build"
GOLDEN = ROOT / "holdout" / "golden.yaml"

SCAN_NOISE = 12.0  # top of the E1 noise range, as in the synthetic set
SCAN_SEED = 7  # the synthetic set uses seed 0


def manifest(item_cls):
    # File names are partly unhelpful on purpose: nothing may be inferred from them.
    return (
        item_cls("supply/contract_D981000001-01.md", "supply/Договор поставки D981000001-01.pdf", "text_pdf"),
        item_cls("supply/supplement_D981000002-01.md", "supply/scan_0412.pdf", "scan_pdf"),
        item_cls("supply/supplement_D981000003-01.md", "supply/ДС новая спецификация.pdf", "text_pdf"),
        item_cls("cleaning/contract_D982000001-01.md", "cleaning/Договор уборки (Самара).docx", "docx"),
        item_cls("cleaning/supplement_D982000002-01.md", "cleaning/ДС окна.pdf", "text_pdf"),
        item_cls("cleaning/supplement_D982000003-01.md", "cleaning/ДС1.pdf", "text_pdf"),
        item_cls(
            "maintenance/contract_D983000001-01.md", "maintenance/Договор ТО кондиционеров.pdf", "text_pdf"
        ),
        item_cls("maintenance/supplement_D983000002-01.md", "maintenance/ДС_финал.pdf", "text_pdf"),
        item_cls("maintenance/supplement_D983000003-01.md", "maintenance/ДС_исправленное.pdf", "text_pdf"),
        item_cls("maintenance/termination_D983000004-01.md", "maintenance/Расторжение.pdf", "text_pdf"),
        item_cls("maintenance/nda_D984000001-01.md", "maintenance/Соглашение NDA.pdf", "text_pdf"),
    )


def sync_golden(ids: dict[str, str]) -> None:
    """Replace stale doc_ids in holdout/golden.yaml; documents are matched by their `file` path."""
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
        print(
            f"warning: {extra.relative_to(ROOT)} is not in the manifest; a run on holdout/build will read it too"
        )
    sync_golden(ids)


if __name__ == "__main__":
    main()
