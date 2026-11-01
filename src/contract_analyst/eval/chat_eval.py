"""Chat evaluation: expected facts present, honest refusals, citations on factual answers."""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import yaml
from pydantic import BaseModel

from contract_analyst.chat.graph import REFUSAL, ChatSession
from contract_analyst.config import Settings
from contract_analyst.kb.store import KnowledgeBase
from contract_analyst.llm.client import LLMClient
from contract_analyst.parsing.patterns import date_spans

_REF = re.compile(r"\[doc:[^\]]+\]")


class ChatQuestion(BaseModel):
    id: str
    question: str
    expected_facts: list[str] = []
    unanswerable: bool = False
    follow_up_of: str | None = None


class ChatScores(BaseModel):
    accuracy: float
    refusal_precision: float
    with_refs_share: float
    answers: dict[str, str] = {}
    invalid: list[str] = []  # answerable questions without expected_facts: asked, not scored for accuracy


def _norm(text: str) -> str:
    return re.sub(r"(?<=\d)[\s ](?=\d{3}\b)", "", " ".join(text.casefold().replace("ё", "е").split()))


def _dates_to_iso(text: str) -> str:
    """Every date written as 01.04.2025 or «1» апреля 2025 г. becomes 2025-04-01, so a fact matches whatever form the
    answer writes it in."""
    out, last = [], 0
    for d, start, end in date_spans(text):
        out += [text[last:start], d.isoformat()]
        last = end
    return "".join(out) + text[last:]


def fact_present(fact: str, answer: str) -> bool:
    return _norm(_dates_to_iso(fact)) in _norm(_dates_to_iso(answer))


def _is_refusal(answer: str) -> bool:
    """The answer is the refusal itself, not an answer that mentions missing data somewhere."""
    return _norm(answer).strip(" .«»\"'") == _norm(REFUSAL).strip(" .«»\"'")


def chat_eval(kb_dir: Path, questions: Path, client: LLMClient, settings: Settings, memory: Path) -> ChatScores:
    items = [ChatQuestion.model_validate(q) for q in yaml.safe_load(questions.read_text(encoding="utf-8"))]
    invalid = [q.id for q in items if not q.unanswerable and not q.expected_facts]
    if invalid:
        warnings.warn(f"answerable questions without expected_facts are not scored: {', '.join(invalid)}",
                      stacklevel=2)
    session = ChatSession(KnowledgeBase(kb_dir), client, settings, memory)
    thread_of: dict[str, str] = {}
    answers, scored, correct, refusals, right_refusals, factual, cited = {}, 0, 0, 0, 0, 0, 0
    for q in items:  # invalid questions are still asked: a follow-up may lean on them
        thread = thread_of.get(q.follow_up_of or "", f"eval-{q.id}")
        thread_of[q.id] = thread
        answer = session.ask(q.question, thread)
        answers[q.id] = answer
        refused = _is_refusal(answer)
        if refused:
            refusals += 1
            right_refusals += q.unanswerable
        else:
            factual += 1
            cited += bool(_REF.search(answer))
        if q.id not in invalid:
            scored += 1
            correct += refused if q.unanswerable else all(fact_present(f, answer) for f in q.expected_facts)
    return ChatScores(accuracy=correct / scored if scored else 1.0,
                      refusal_precision=right_refusals / refusals if refusals else 1.0,
                      with_refs_share=cited / factual if factual else 1.0, answers=answers, invalid=invalid)
