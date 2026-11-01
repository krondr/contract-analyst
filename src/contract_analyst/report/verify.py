"""Code-level claim check for LLM-written paragraphs: a sentence with a number needs a valid source ref."""

from __future__ import annotations

import re

_BLOCK = re.compile(r"<!-- llm -->(.*?)<!-- /llm -->", re.DOTALL)
_REF = re.compile(r"\[doc:[^\]\s]+\]")
# A possible sentence end: terminal punctuation with the closing quotes and references that belong to the
# sentence, then whitespace; or a line break. A period inside "3.1" or "01.04.2025" is not followed by a space.
_END = re.compile(r"(?P<mark>[.!?…]+)[»\"”)]*(?:\s*\[doc:[^\]\s]+\])*(?P<close>[.!?…]?)(?=\s|$)|\n")
# Abbreviations after which a sentence goes on even before a capital: "ст. 450 ГК", "ул. Ленина", "т.е. Договор".
# Single letters are handled apart: initials, "п.", "ч.", "г. Тверь".
_INNER = {"пп", "ст", "гл", "разд", "подп", "абз", "прил", "см", "им", "ул", "обл", "пр", "пер", "кв", "стр", "т.е",
          "т.ч", "т.к", "напр", "тел", "рис", "табл", "ред", "№"}


def _ends(text: str, start: int, m: re.Match[str]) -> bool:
    if m.group(0) == "\n":
        return True
    following = text[m.end():].lstrip().lstrip("«\"„“*")
    if not following:
        return True
    if not following[0].isupper():  # "руб. в месяц", "п. 3.1", "т.е. 1 500"
        return False
    if m.group("mark") != "." or m.group("close"):
        return True
    words = text[start:m.start()].split()
    word = words[-1].lstrip("«\"(").lower() if words else ""
    if word in _INNER or (word.isdigit() and len(words) == 1):  # list numbering: "1. Ставка ..."
        return False
    if len(word) == 1 and word.isalpha():  # "г." ends a sentence only as "год": "в 2025 г. Срок ..."
        return word == "г" and len(words) > 1 and words[-2][-1].isdigit()
    return True


def split_sentences(text: str) -> list[str]:
    """Split into sentences without losing characters: "".join(parts) == text."""
    parts, start = [], 0
    for m in _END.finditer(text):
        if m.start() < start or not _ends(text, start, m):
            continue
        end = len(text) - len(text[m.end():].lstrip())
        parts.append(text[start:end])
        start = end
    return parts + [text[start:]] if start < len(text) else parts


def _check_block(text: str, valid_refs: set[str]) -> tuple[str, int]:
    kept, removed = [], 0
    for part in split_sentences(text.strip()):
        sentence = part.strip()
        refs = [r[1:-1] for r in _REF.findall(sentence)]
        bare = _REF.sub("", sentence)
        if any(r not in valid_refs for r in refs) or (any(ch.isdigit() for ch in bare) and not refs):
            removed += 1
            continue
        kept.append(part)
    return "".join(kept).strip(), removed


def verify_claims(markdown: str, valid_refs: set[str]) -> tuple[str, int]:
    total = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal total
        text, removed = _check_block(match.group(1), valid_refs)
        total += removed
        return f"<!-- llm -->{text}<!-- /llm -->"

    return _BLOCK.sub(replace, markdown), total
