"""Chat tools: read-only queries over the knowledge base; every fact comes back with its source reference."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

from pydantic import BaseModel, Field

from contract_analyst.kb.store import KnowledgeBase
from contract_analyst.labels import STATUS_RU
from contract_analyst.models import Source, ValueStatus
from contract_analyst.report.render import money


def _ref(source: Source | None) -> str:
    return f"[doc:{source.doc_id}:p{source.page}]" if source else ""


class ChatArgs(BaseModel):
    thought: str = Field("", description="Что ищешь и зачем")


class ContractArgs(ChatArgs):
    contract_number: str


class TermsArgs(ContractArgs):
    as_of: date | None = Field(None, description="Дата, на которую нужны условия")


class HistoryArgs(ContractArgs):
    term: str = Field(..., description="Ключ условия, например validity или payment_terms")


class SearchArgs(ChatArgs):
    query: str
    contract_number: str | None = None


class IssuesArgs(ChatArgs):
    contract_number: str | None = None


def contract_terms(kb: KnowledgeBase, a: TermsArgs) -> dict:
    c = kb.get_contract_terms(a.contract_number, a.as_of)
    if c is None:
        return {"error": f"договора {a.contract_number} нет в базе"}
    return {"contract": c.contract_number, "as_of": c.as_of, "status": STATUS_RU.get(c.status, c.status),
            "status_since": c.status_since,
            "terms": {k: f"{v.current.value} {_ref(v.current.source)}" for k, v in c.terms.items()
                      if v.current.status == ValueStatus.ACCEPTED},
            "rates": [f"{r.label}: {money(r.amount)} руб. ({r.vat}) с {r.valid_from or '—'} по {r.valid_to or '—'} "
                      f"{_ref(r.source)}" for r in c.rates],
            "planned_rates": [f"{r.label}: {money(r.amount)} руб. с {r.valid_from} {_ref(r.source)}" for r in c.planned_rates],
            "documents": [f"{e.number or e.doc_id} с {e.when or '—'}: {e.summary}{' (отменён)' if e.invalidated else ''}"
                          for e in c.timeline]}


def term_history(kb: KnowledgeBase, a: HistoryArgs) -> dict:
    steps = kb.get_term_history(a.contract_number, a.term)
    return {"term": a.term, "changes": [f"{s.value} (с {s.effective_from or '—'}) {_ref(s.source)}" for s in steps]}


def search(kb: KnowledgeBase, a: SearchArgs) -> dict:
    return {"hits": [f"{h.contract_number or '—'} п. {h.clause_id}: {h.snippet[:240]} [doc:{h.doc_id}:p{h.page}]"
                     for h in kb.search_clauses(a.query, a.contract_number)]}


def issues(kb: KnowledgeBase, a: IssuesArgs) -> dict:
    return {"issues": [f"{i.code}: {i.message} {' '.join(_ref(s) for s in i.sources)}"
                       for i in kb.get_issues(a.contract_number)][:20]}


@dataclass(frozen=True)
class ChatTool:
    name: str
    description: str
    args_model: type[ChatArgs]
    fn: Callable[[KnowledgeBase, ChatArgs], dict]


CHAT_TOOLS = {t.name: t for t in [
    ChatTool("list_contracts", "Список договоров в базе: номер, статус, кластер, число документов.", ChatArgs,
             lambda kb, a: {"contracts": kb.list_contracts()}),
    ChatTool("get_contract_terms", "Действующие условия, ставки и документы договора на дату.", TermsArgs, contract_terms),
    ChatTool("get_term_history", "История изменения одного условия договора.", HistoryArgs, term_history),
    ChatTool("search_clauses", "Поиск по тексту пунктов договоров (с учётом словоформ).", SearchArgs, search),
    ChatTool("get_issues", "Замечания и противоречия по договору: конфликты дат, отмены, пропавшие документы.",
             IssuesArgs, issues),
]}


def chat_tool_schemas() -> list[dict]:
    return [{"type": "function", "function": {"name": t.name, "description": t.description,
                                              "parameters": t.args_model.model_json_schema()}} for t in CHAT_TOOLS.values()]


def run_chat_tool(kb: KnowledgeBase, name: str, args: dict) -> str:
    tool = CHAT_TOOLS.get(name)
    if tool is None:
        return json.dumps({"error": f"нет инструмента {name}"}, ensure_ascii=False)
    try:
        parsed = tool.args_model.model_validate(args)
    except ValueError as exc:
        return json.dumps({"error": f"неверные аргументы: {str(exc)[:200]}"}, ensure_ascii=False)
    return json.dumps(tool.fn(kb, parsed), ensure_ascii=False, default=str)[:4000]
