"""Chat evaluation: expected facts present, honest refusals, citations on factual answers."""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from pydantic import BaseModel

from contract_analyst.chat.graph import REFUSAL, ChatSession
from contract_analyst.config import Settings
from contract_analyst.kb.store import KnowledgeBase
from contract_analyst.llm.client import LLMClient

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


def _norm(text: str) -> str:
    return re.sub(r"(?<=\d)[\s ](?=\d{3}\b)", "", " ".join(text.casefold().replace("ё", "е").split()))


def chat_eval(kb_dir: Path, questions: Path, client: LLMClient, settings: Settings, memory: Path) -> ChatScores:
    items = [ChatQuestion.model_validate(q) for q in yaml.safe_load(questions.read_text(encoding="utf-8"))]
    session = ChatSession(KnowledgeBase(kb_dir), client, settings, memory)
    thread_of: dict[str, str] = {}
    answers, correct, refusals, right_refusals, factual, cited = {}, 0, 0, 0, 0, 0
    for q in items:
        thread = thread_of.get(q.follow_up_of or "", f"eval-{q.id}")
        thread_of[q.id] = thread
        answer = session.ask(q.question, thread)
        answers[q.id] = answer
        refused = REFUSAL.casefold().rstrip(".") in answer.casefold() or "нет данных" in answer.casefold()
        if refused:
            refusals += 1
            right_refusals += q.unanswerable
        else:
            factual += 1
            cited += bool(_REF.search(answer))
        ok = refused if q.unanswerable else all(_norm(f) in _norm(answer) for f in q.expected_facts)
        correct += ok
    return ChatScores(accuracy=correct / len(items) if items else 1.0,
                      refusal_precision=right_refusals / refusals if refusals else 1.0,
                      with_refs_share=cited / factual if factual else 1.0, answers=answers)
