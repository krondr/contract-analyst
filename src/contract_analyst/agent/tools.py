"""Agent tools: thin, validated wrappers over steps. Every call carries the agent's thought."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from contract_analyst.agent import steps
from contract_analyst.agent.context import INPUTS, PARTS, RunContext
from contract_analyst.agent.guards import Guard
from contract_analyst.llm.client import ToolCall
from contract_analyst.models import Decision, ToolResult


class ThoughtArgs(BaseModel):
    thought: str = Field("", description="Что ты делаешь этим вызовом и зачем — одно-два предложения")


class ListInputsArgs(ThoughtArgs):
    pass


class ReadDocumentArgs(ThoughtArgs):
    file_name: str = Field(..., description="Имя файла из описи")


class RerecognizeArgs(ThoughtArgs):
    doc_id: str
    pages: list[int] = Field(..., description="Номера страниц с низкой уверенностью OCR")
    strategy: Literal["higher_dpi", "rotate", "alt_engine", "preprocess"]


class AnalyzeDocumentArgs(ThoughtArgs):
    doc_id: str
    groups: list[str] | None = Field(None, description="Группы полей; по умолчанию все")


class BuildChainsArgs(ThoughtArgs):
    pass


class ConsolidateArgs(ThoughtArgs):
    contract_number: str | None = Field(None, description="Номер договора; по умолчанию все")
    as_of: date | None = Field(None, description="Дата свёртки; по умолчанию дата запуска")


class ClusterArgs(ThoughtArgs):
    pass


class RecordDecisionArgs(ThoughtArgs):
    issue_code: str
    contract_number: str | None = None
    decision: str
    rationale: str


class WriteReportArgs(ThoughtArgs):
    pass


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_model: type[ThoughtArgs]
    fn: Callable[[RunContext, ThoughtArgs], ToolResult]
    reads: tuple[str, ...] = ()  # state parts the result depends on: a repeat counts only while they are unchanged


def _list_inputs(ctx: RunContext) -> ToolResult:
    if ctx.agent_reads and not ctx.documents:
        return steps.list_only(ctx)
    if not ctx.documents:
        return steps.ingest_all(ctx)
    return ToolResult(status="ok", data=steps.fit({"files": len(ctx.inventory), "unprocessed": ctx.unprocessed,
                                                   "documents": [steps.doc_line(ctx, d) for d in ctx.documents.values()]}))


TOOLS: dict[str, ToolSpec] = {spec.name: spec for spec in [
    ToolSpec("list_inputs", "Опись входных файлов: формат, способ чтения страниц, найденные номера, качество.",
             ListInputsArgs, lambda ctx, a: _list_inputs(ctx), ("documents",)),
    ToolSpec("read_document", "Прочитать файл (текстовый слой или OCR) — для файла, которого ещё нет в описи.",
             ReadDocumentArgs, lambda ctx, a: steps.read_one(ctx, a.file_name)),
    ToolSpec("rerecognize", "Перераспознать страницы скана другими настройками, если уверенность OCR низкая.",
             RerecognizeArgs, lambda ctx, a: steps.rerecognize(ctx, a.doc_id, a.pages, a.strategy)),
    ToolSpec("analyze_document", "Разобрать документ: вид, реквизиты, условия, ставки, операции изменения. "
             "Значения проходят шлюз проверок.", AnalyzeDocumentArgs,
             lambda ctx, a: steps.analyze_docs(ctx, [a.doc_id], a.groups), ("documents",)),
    ToolSpec("build_chains", "Собрать цепочки договоров из разобранных документов: порядок, отмены, конфликты.",
             BuildChainsArgs, lambda ctx, a: steps.assemble_chains(ctx), INPUTS["chains"]),
    ToolSpec("consolidate", "Свернуть действующие условия договора на дату с учётом допсоглашений.",
             ConsolidateArgs, lambda ctx, a: steps.consolidate_all(ctx, a.contract_number, a.as_of),
             INPUTS["consolidated"]),
    ToolSpec("cluster_and_compare", "Кластеризовать договоры по предмету и сравнить договоры внутри кластера.",
             ClusterArgs, lambda ctx, a: steps.cluster_all(ctx), INPUTS["clusters"]),
    ToolSpec("record_decision", "Записать твоё решение по противоречию (оно попадёт в отчёт).", RecordDecisionArgs,
             lambda ctx, a: steps.record_decision(ctx, Decision(issue_code=a.issue_code,
                                                                contract_number=a.contract_number,
                                                                decision=a.decision, rationale=a.rationale))),
    ToolSpec("write_report", "Собрать итоговый отчёт. Вызывай последним.", WriteReportArgs,
             lambda ctx, a: steps.write_report(ctx), PARTS),
]}


def tool_schemas(names: list[str] | None = None) -> list[dict]:
    return [{"type": "function", "function": {"name": s.name, "description": s.description,
                                              "parameters": s.args_model.model_json_schema()}}
            for s in TOOLS.values() if names is None or s.name in names]


_THOUGHT_PREFIX = re.compile(r"^\s*(?:мысль|thought)\s*[:\-—]\s*", re.IGNORECASE)


_REASONING_TAIL = 600


def _thought(args: ThoughtArgs, message_text: str, reasoning: str = "") -> tuple[str, str]:
    """The model's own words: the thought argument, else the text it wrote next to the calls, else the end of its
    thinking, where it settles on the calls (the whole thinking is in the executor's llm_call event)."""
    if args.thought.strip():
        return args.thought.strip(), "argument"
    text = _THOUGHT_PREFIX.sub("", message_text.strip())
    if text:
        return text[:500], "message"
    reasoning = reasoning.strip()
    if reasoning:
        return (reasoning if len(reasoning) <= _REASONING_TAIL else "…" + reasoning[-_REASONING_TAIL:]), "reasoning"
    return "", "missing"


def dispatch(ctx: RunContext, call: ToolCall, guard: Guard, node: str = "tools", message_text: str = "",
             reasoning: str = "") -> ToolResult:
    spec = TOOLS.get(call.name)
    if spec is None:
        result = ToolResult(status="error", error_type="unknown_tool", hint=f"Нет инструмента {call.name}")
        ctx.trace.event("error", node, tool=call.name, message="unknown tool")
        return result
    if (budget := guard.check_budget()) is not None:
        ctx.trace.event("error", node, tool=call.name, message=budget.hint)
        return budget
    try:
        args = spec.args_model.model_validate(call.args)
    except ValidationError as exc:
        ctx.trace.event("error", node, tool=call.name, message="bad arguments")
        return ToolResult(status="error", error_type="bad_args", hint=f"Неверные аргументы: {str(exc)[:300]}")
    thought, source = _thought(args, message_text, reasoning)
    ctx.trace.event("thought", node, tool=call.name, thought=thought, thought_source=source)
    args_key = json.dumps(args.model_dump(exclude={"thought"}), sort_keys=True, default=str)
    guard.steps += 1
    if guard.seen((call.name, args_key, ctx.revisions(spec.reads))):
        result = ToolResult(status="error", error_type="repeat_call",
                            hint="Этот вызов уже был, и данные с тех пор не менялись: смени аргументы "
                                 "или переходи к следующему шагу")
    else:
        ctx.trace.event("tool_call", node, tool=call.name, thought=thought, thought_source=source,
                        args=args.model_dump(exclude={"thought"}, mode="json"))
        try:
            result = spec.fn(ctx, args)
        except Exception as exc:  # noqa: BLE001 — a failing tool must become an observation, not a crash
            ctx.trace.event("error", node, tool=call.name, message=f"{type(exc).__name__}: {exc}"[:300])
            result = ToolResult(status="error", error_type="exception", hint=f"{type(exc).__name__}: {exc}"[:300])
        # What the call changed itself (list_inputs reading the files) does not make the same call new.
        guard.seen((call.name, args_key, ctx.revisions(spec.reads)))
    ctx.trace.event("tool_result", node, tool=call.name, status=result.status, error_type=result.error_type,
                    data=json.dumps(result.data, ensure_ascii=False, default=str)[:500])
    return result
