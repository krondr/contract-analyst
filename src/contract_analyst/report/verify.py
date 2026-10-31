"""Code-level claim check for LLM-written paragraphs: a sentence with a number needs a valid source ref."""

from __future__ import annotations

import re

_BLOCK = re.compile(r"<!-- llm -->(.*?)<!-- /llm -->", re.DOTALL)
_REF = re.compile(r"\[doc:[^\]\s]+\]")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _check_block(text: str, valid_refs: set[str]) -> tuple[str, int]:
    kept, removed = [], 0
    for sentence in (s for s in _SENTENCE.split(text.strip()) if s):
        refs = [r[1:-1] for r in _REF.findall(sentence)]
        bare = _REF.sub("", sentence)
        if any(r not in valid_refs for r in refs) or (any(ch.isdigit() for ch in bare) and not refs):
            removed += 1
            continue
        kept.append(sentence)
    return " ".join(kept), removed


def verify_claims(markdown: str, valid_refs: set[str]) -> tuple[str, int]:
    total = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal total
        text, removed = _check_block(match.group(1), valid_refs)
        total += removed
        return f"<!-- llm -->{text}<!-- /llm -->"

    return _BLOCK.sub(replace, markdown), total
