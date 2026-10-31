"""Russian lemmatization for full-text search over clauses."""

from __future__ import annotations

import re
from functools import lru_cache

import pymorphy3

_TOKEN = re.compile(r"D\d{9}(?:-\d{2})?|[0-9]+(?:[.,][0-9]+)*|[A-Za-zА-Яа-яЁё]+(?:-[A-Za-zА-Яа-яЁё]+)*")


@lru_cache(maxsize=1)
def _morph() -> pymorphy3.MorphAnalyzer:
    return pymorphy3.MorphAnalyzer()


@lru_cache(maxsize=50_000)
def _lemma(word: str) -> str:
    if not word[0].isalpha() or word.startswith("D") and word[1:2].isdigit():
        return word.lower()
    return _morph().parse(word.lower())[0].normal_form


def lemmatize(text: str) -> str:
    return " ".join(_lemma(t) for t in _TOKEN.findall(text))
