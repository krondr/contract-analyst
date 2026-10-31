"""Character and word error rates; whitespace differences (line breaks) are not counted."""

from __future__ import annotations

from rapidfuzz.distance import Levenshtein


def _norm(text: str) -> str:
    return " ".join(text.split())


def cer(hyp: str, ref: str) -> float:
    ref_n = _norm(ref)
    return Levenshtein.distance(_norm(hyp), ref_n) / max(1, len(ref_n))


def wer(hyp: str, ref: str) -> float:
    ref_w = ref.split()
    return Levenshtein.distance(hyp.split(), ref_w) / max(1, len(ref_w))
