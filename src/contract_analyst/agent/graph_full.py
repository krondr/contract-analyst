"""Variant C: a full agent — no code ingest; the agent lists and reads every file itself."""

from __future__ import annotations

from contract_analyst.agent.context import RunContext
from contract_analyst.agent.graph_hybrid import build_hybrid_graph


def build_full_agent_graph(ctx: RunContext):
    ctx.agent_reads = True
    return build_hybrid_graph(ctx, with_ingest=False)
