"""Variant A (reference point): the same steps in a fixed order, no planning, no agent decisions."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from contract_analyst.agent import steps
from contract_analyst.agent.context import RunContext
from contract_analyst.agent.state import AgentState


def build_pipeline_graph(ctx: RunContext):
    from contract_analyst.agent.graph_hybrid import kb_check, reflect_checklist

    def step(name, fn):
        def node(state: AgentState) -> dict:
            result = fn()
            ctx.trace.event("step", "pipeline", step=name, status=result.status)
            return {}
        return node

    def report(state: AgentState) -> dict:
        steps.write_report(ctx)  # writes the knowledge base the checklist reads
        ctx.reflection = reflect_checklist(ctx) + [kb_check(ctx)]
        result = steps.write_report(ctx)
        ctx.trace.event("step", "pipeline", step="write_report", status=result.status)
        ctx.trace.event("final", "pipeline", summary=str(ctx.report_path))
        return {}

    order = [("ingest", lambda: steps.ingest_all(ctx) if not ctx.documents else steps.ToolResult(status="ok")),
             ("analyze", lambda: steps.analyze_docs(ctx, [d for d in ctx.documents
                                                          if ctx.only_doc_ids is None or d in ctx.only_doc_ids])),
             ("assemble", lambda: steps.assemble_chains(ctx)), ("consolidate", lambda: steps.consolidate_all(ctx)),
             ("cluster", lambda: steps.cluster_all(ctx))]
    graph = StateGraph(AgentState)
    for name, fn in order:
        graph.add_node(name, step(name, fn))
    graph.add_node("report", report)
    previous = START
    for name, _ in order:
        graph.add_edge(previous, name)
        previous = name
    graph.add_edge(previous, "report")
    graph.add_edge("report", END)
    return graph.compile()
