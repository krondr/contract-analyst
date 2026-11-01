"""Text normalization applied to every page before any parsing or LLM call."""

from __future__ import annotations

import re
import unicodedata

_LATIN_TO_CYRILLIC = str.maketrans("ABCEHKMOPTXaceopxy", "АВСЕНКМОРТХасеорху")
_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё]+")
_HAS_CYRILLIC = re.compile(r"[А-Яа-яЁё]")
_HAS_LATIN = re.compile(r"[A-Za-z]")
_HYPHEN_BREAK = re.compile(r"([A-Za-zА-Яа-яЁё])-\n([a-zа-яё])")
_SPACES = re.compile(r"[ \t ]+")

# Blanks, dashes, empty quotes, and letter masks a template leaves instead of a value («ХХХХ рублей», «XXX руб.»).
_REDACTION_RE = re.compile(r"_{3,}|(?:-\s){2,}-|«\s*»|\"\s*\"|(?<![А-Яа-яЁёA-Za-z])[ХхXx]{3,}(?:[.,][ХхXx]{2,})?(?![А-Яа-яЁёA-Za-z])")


def _fix_homoglyphs(match: re.Match[str]) -> str:
    word = match.group(0)
    if _HAS_CYRILLIC.search(word) and _HAS_LATIN.search(word):
        return word.translate(_LATIN_TO_CYRILLIC)
    return word


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = _WORD_RE.sub(_fix_homoglyphs, text)
    lines = [_SPACES.sub(" ", line).strip() for line in text.split("\n")]
    return "\n".join(lines)


def is_redacted(fragment: str) -> bool:
    return bool(_REDACTION_RE.search(fragment))
