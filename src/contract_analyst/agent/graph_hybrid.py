"""Variant B (hybrid): code reads every file first, then the LLM agent plans, acts through tools and reflects."""

from __future__ import annotations

import json

import httpx
from langgraph.graph import END, START, StateGraph

from contract_analyst.agent import steps
from contract_analyst.agent.context import RunContext
from contract_analyst.agent.guards import Guard
from contract_analyst.agent.state import AgentState, PlanOut, ReflectionOut
from contract_analyst.agent.tools import dispatch, tool_schemas
from contract_analyst.llm.client import ToolCall
from contract_analyst.llm.prompts import load_prompt
from contract_analyst.llm.structured import StructuredFailure, call_structured
from contract_analyst.report.render import ReflectionItem

MANDATORY = ["Каждый файл разобран или отклонён с причиной", "Цепочки договоров собраны",
             "Условия свёрнуты на дату анализа", "Кластеры и сравнение внутри кластеров готовы"]
REQUIRED_HEADINGS = ["## 1. Сводка запуска", "## 2. Опись документов", "## 3. Кластеры", "## 4. Договоры",
                     "## 5. Отказы и противоречия", "## 6. Не обработано", "## 7. Reflection"]


def safe_structured(ctx: RunContext, node: str, messages, schema):
    """Structured call where a timeout counts as a failed answer instead of crashing the run."""
    try:
        return call_structured(ctx.client, messages, schema)
    except (TimeoutError, httpx.TimeoutException) as exc:
        ctx.trace.event("error", node, message=f"LLM timeout: {exc}")
        return StructuredFailure(errors=[f"timeout: {exc}"], raw=[])


def reflect_checklist(ctx: RunContext) -> list[ReflectionItem]:
    scope = [d for d in ctx.documents.values() if ctx.only_doc_ids is None or d.doc_id in ctx.only_doc_ids]
    missing = [d.file.name for d in scope if d.doc_id not in ctx.records]
    unconsolidated = [c.contract_number for c in ctx.chains if c.contract_number not in ctx.consolidated]
    refused = sum(1 for r in ctx.records.values() for v in [r.number, r.doc_date, r.parent, r.effective_from,
                                                            *r.terms.values(), *r.rates, *r.ops]
                  if v.status.value in ("rejected", "conflict", "redacted"))
    stale = " — устарели: исходные данные изменились после сборки"
    chains_stale, consolidated_stale = ctx.stale("chains"), ctx.stale("consolidated")
    clusters_stale = ctx.stale("clusters")
    return [
        ReflectionItem(item=MANDATORY[0], ok=not missing,
                       note=f"не разобраны: {', '.join(missing)}" if missing
                       else f"разобрано {len(ctx.records)}, не обработано файлов: {len(ctx.unprocessed)}"),
        ReflectionItem(item=MANDATORY[1], ok=not ctx.records or (bool(ctx.chains) and not chains_stale),
                       note=f"цепочек: {len(ctx.chains)}" + (stale if ctx.chains and chains_stale else "")),
        ReflectionItem(item=MANDATORY[2], ok=bool(ctx.chains) and not unconsolidated and not consolidated_stale,
                       note=f"без свёртки: {', '.join(unconsolidated)}" if unconsolidated
                       else f"на {ctx.as_of}" + (stale if consolidated_stale else "")),
        ReflectionItem(item=MANDATORY[3], ok=not ctx.consolidated or (bool(ctx.clusters) and not clusters_stale),
                       note=f"кластеров: {len(ctx.clusters)}" + (stale if ctx.clusters and clusters_stale else "")),
        ReflectionItem(item="Непроверенные значения не записаны в базу знаний", ok=True,
                       note=f"отклонено или спорно: {refused} — показаны в разделе 5"),
    ]


def _inventory_text(ctx: RunContext) -> str:
    lines = [f"Дата анализа: {ctx.as_of}. Файлов: {len(ctx.inventory)}, прочитано: {len(ctx.documents)}."]
    for d in ctx.documents.values():
        line = steps.doc_line(ctx, d)
        lines.append(f"- doc_id={line['doc_id']} файл=«{line['file']}» стр={line['pages']} чтение={line['methods']} "
                     f"качество={line['status']} номер={line['number']} ссылается_на={line['parent']}"
                     + (f" страницы_с_низкой_уверенностью={line['low_conf_pages']}" if line["low_conf_pages"] else ""))
    lines += [f"- не обработан: {u['file']} — {u['reason']}" for u in ctx.unprocessed]
    return "\n".join(lines)


def build_hybrid_graph(ctx: RunContext, with_ingest: bool = True):
    guard = Guard(ctx.settings.max_steps, ctx.settings.max_seconds)

    def ingest(state: AgentState) -> dict:
        if with_ingest:
            steps.ingest_all(ctx)
        return {}

    def planner(state: AgentState) -> dict:
        inventory = _inventory_text(ctx) if with_ingest else "Файлы ещё не прочитаны: начни с list_inputs."
        prompt = load_prompt("planner")
        plan = safe_structured(ctx, "planner", [{"role": "system", "content": prompt.render(as_of=ctx.as_of)},
                                                {"role": "user", "content": inventory}], PlanOut)
        items = (plan.items if not isinstance(plan, StructuredFailure) else []) + MANDATORY
        ctx.trace.event("plan", "planner", items=items)
        system = load_prompt("executor").render(as_of=ctx.as_of, plan=items, with_ingest=with_ingest)
        return {"plan": items, "messages": [{"role": "system", "content": system},
                                            {"role": "user", "content": inventory}]}

    def executor(state: AgentState) -> dict:
        try:
            turn = ctx.client.chat(state["messages"], tool_schemas(), think=ctx.settings.agent_thinking)
        except (TimeoutError, httpx.TimeoutException) as exc:
            ctx.trace.event("error", "executor", message=f"LLM timeout: {exc}")
            return {"messages": [{"role": "assistant", "content": "", "tool_calls": []}]}
        ctx.trace.event("llm_call", "executor", content=turn.content[:300], reasoning=turn.reasoning,
                        tool_calls=[c.name for c in turn.tool_calls])
        message = {"role": "assistant", "content": turn.content, "tool_calls": [c.model_dump() for c in turn.tool_calls]}
        return {"messages": [message | ({"reasoning": turn.reasoning} if turn.reasoning else {})]}

    def after_executor(state: AgentState) -> str:
        return "tools" if state["messages"][-1].get("tool_calls") and not guard.exhausted else "reflector"

    def tools(state: AgentState) -> dict:
        out = []
        message_text = state["messages"][-1].get("content") or ""
        reasoning = state["messages"][-1].get("reasoning") or ""
        for raw in state["messages"][-1]["tool_calls"]:
            call = ToolCall.model_validate(raw)
            result = dispatch(ctx, call, guard, message_text=message_text, reasoning=reasoning)
            out.append({"role": "tool", "tool_call_id": call.id, "name": call.name,
                        "content": json.dumps(result.model_dump(), ensure_ascii=False, default=str)})
        return {"messages": out}

    def after_tools(state: AgentState) -> str:
        return "reflector" if guard.exhausted else "executor"

    def reflector(state: AgentState) -> dict:
        items = reflect_checklist(ctx)
        prompt = load_prompt("reflector")
        summary = "\n".join(f"- {'OK' if i.ok else 'НЕТ'}: {i.item} ({i.note})" for i in items)
        verdict = safe_structured(ctx, "reflector", [
            {"role": "system", "content": prompt.render(plan=state["plan"])},
            {"role": "user", "content": f"Проверки кодом:\n{summary}"}], ReflectionOut)
        llm_gaps = [] if isinstance(verdict, StructuredFailure) else verdict.gaps
        gaps = [f"{i.item}: {i.note}" for i in items if not i.ok] + llm_gaps
        round_no = state["reflection_round"] + 1
        ctx.trace.event("reflection", "reflector", round=round_no, items=[i.model_dump() for i in items], gaps=gaps)
        ctx.reflection = items
        update: dict = {"reflection_round": round_no, "gaps": gaps}
        if gaps and round_no < ctx.settings.max_reflection_rounds and not guard.exhausted:
            update["messages"] = [{"role": "user", "content": "Рефлексия: цель ещё не достигнута. Доделай:\n"
                                   + "\n".join(f"- {g}" for g in gaps)}]
        return update

    def after_reflector(state: AgentState) -> str:
        loop = state["gaps"] and state["reflection_round"] < ctx.settings.max_reflection_rounds and not guard.exhausted
        return "executor" if loop else "reporter"

    def reporter(state: AgentState) -> dict:
        complete_with_code(ctx)
        ctx.reflection = reflect_checklist(ctx) + [ReflectionItem(
            item="Шаги, выполненные кодом за агента", ok=not ctx.safety_net,
            note=", ".join(ctx.safety_net) or "нет — агент выполнил план сам")]
        steps.write_report(ctx)
        text = ctx.report_path.read_text(encoding="utf-8")
        missing = [h for h in REQUIRED_HEADINGS if h not in text]
        ctx.reflection.append(ReflectionItem(item="Все разделы отчёта на месте", ok=not missing,
                                             note=", ".join(missing) or "8 разделов"))
        steps.write_report(ctx)
        ctx.trace.event("final", "reporter", summary=str(ctx.report_path))
        return {}

    graph = StateGraph(AgentState)
    for name, fn in [("ingest", ingest), ("planner", planner), ("executor", executor), ("tools", tools),
                     ("reflector", reflector), ("reporter", reporter)]:
        graph.add_node(name, fn)
    graph.add_edge(START, "ingest")
    graph.add_edge("ingest", "planner")
    graph.add_edge("planner", "executor")
    graph.add_conditional_edges("executor", after_executor, ["tools", "reflector"])
    graph.add_conditional_edges("tools", after_tools, ["executor", "reflector"])
    graph.add_conditional_edges("reflector", after_reflector, ["executor", "reporter"])
    graph.add_edge("reporter", END)
    return graph.compile()


def complete_with_code(ctx: RunContext) -> None:
    """Safety net: whatever the agent left undone is finished by code and recorded as such."""
    if not ctx.documents and not ctx.inventory:
        steps.ingest_all(ctx)
        ctx.safety_net.append("ingest_all")
    scope = [d for d in ctx.documents if (ctx.only_doc_ids is None or d in ctx.only_doc_ids) and d not in ctx.records]
    if scope:
        steps.analyze_docs(ctx, scope)
        ctx.safety_net.append("analyze_docs")
    # Each part is redone when its inputs changed after it was built, so a rebuild cascades:
    # new records -> chains -> consolidation -> clusters.
    if ctx.records and ctx.stale("chains"):
        steps.assemble_chains(ctx)
        ctx.safety_net.append("assemble_chains")
    if ctx.chains and ctx.stale("consolidated"):
        steps.consolidate_all(ctx)
        ctx.safety_net.append("consolidate_all")
    if ctx.consolidated and ctx.stale("clusters"):
        steps.cluster_all(ctx)
        ctx.safety_net.append("cluster_all")
