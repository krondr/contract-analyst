"""Input inventory: which files we got, their format, hashes and duplicates."""

from __future__ import annotations

import hashlib
import unicodedata
from pathlib import Path

from contract_analyst.models import DocFormat, InputFile

_FORMATS = {".pdf": DocFormat.PDF, ".docx": DocFormat.DOCX, ".jpg": DocFormat.IMAGE, ".jpeg": DocFormat.IMAGE,
            ".png": DocFormat.IMAGE, ".tif": DocFormat.IMAGE, ".tiff": DocFormat.IMAGE}
_SYSTEM_NAMES = {"Thumbs.db", "desktop.ini"}


def _skipped(rel: Path) -> bool:
    return any(part.startswith(".") for part in rel.parts) or rel.name.startswith("~$") or rel.name in _SYSTEM_NAMES


def list_inputs(root: Path) -> list[InputFile]:
    files: list[InputFile] = []
    first_by_hash: dict[str, str] = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if _skipped(path.relative_to(root)):
            continue
        data = path.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        files.append(InputFile(
            path=str(path), name=unicodedata.normalize("NFC", path.name), sha256=sha,
            format=_FORMATS.get(path.suffix.lower(), DocFormat.UNSUPPORTED), size_bytes=len(data),
            duplicate_of=first_by_hash.get(sha)))
        first_by_hash.setdefault(sha, str(path))
    return files
