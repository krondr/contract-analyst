"""One analysis run: build the context, run the chosen orchestration variant, write metrics."""

from __future__ import annotations

import json
import time
from datetime import date, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from contract_analyst.agent.context import RunContext
from contract_analyst.agent.trace import TraceLogger
from contract_analyst.config import Settings
from contract_analyst.llm.client import LLMClient
from contract_analyst.llm.factory import get_llm_client
from contract_analyst.ocr import OcrEngine, get_ocr_engine
from contract_analyst.understand.understand import prompt_versions

DEFAULT_LLM_CACHE = Path("data/cache/llm.sqlite")
DEFAULT_READ_CACHE = Path("data/cache/read")


class RunResult(BaseModel):
    run_dir: Path
    report_path: Path | None
    kb_dir: Path
    trace_path: Path
    variant: str
    seconds: float
    llm_usage: dict


def _run_dir(out_dir: Path, variant: str) -> Path:
    base = out_dir / f"{datetime.now():%Y%m%d-%H%M%S}-{variant}"
    path, n = base, 1
    while path.exists():
        n += 1
        path = base.with_name(f"{base.name}-{n}")
    return path


def _restrict_to_chain(ctx: RunContext, number: str) -> None:
    ctx.only_doc_ids = {doc_id for doc_id, h in ctx.headers.items() if number in (h.number, h.parent_number)}


def run_analysis(input_dir: Path, settings: Settings, as_of: date, variant: Literal["A", "B", "C"] = "B",
                 only_chain: str | None = None, only_doc: Path | None = None, out_dir: Path = Path("runs"),
                 client: LLMClient | None = None, ocr_engine: OcrEngine | None = None,
                 read_cache: Path | None = DEFAULT_READ_CACHE, llm_cache: Path = DEFAULT_LLM_CACHE,
                 console: bool = True) -> RunResult:
    run_dir = _run_dir(out_dir, variant)
    if client is None:
        client = get_llm_client(settings, llm_cache)
        if hasattr(client, "check_available"):
            client.check_available()
    ocr_engine = ocr_engine or get_ocr_engine(settings.ocr_engine)
    trace = TraceLogger(run_dir, run_dir.name, console=console)
    ctx = RunContext(settings=settings, client=client, ocr_engine=ocr_engine, trace=trace, input_dir=input_dir,
                     run_dir=run_dir, as_of=as_of, read_cache=read_cache,
                     only_paths={str(only_doc)} if only_doc else None)
    started = time.monotonic()
    if variant == "B":
        from contract_analyst.agent.graph_hybrid import build_hybrid_graph

        graph = build_hybrid_graph(ctx)
        if only_chain:
            from contract_analyst.agent import steps

            steps.ingest_all(ctx)
            _restrict_to_chain(ctx, only_chain)
            graph = build_hybrid_graph(ctx, with_ingest=False)
    else:
        raise NotImplementedError(f"variant {variant} arrives with experiment E4")
    graph.invoke({"messages": [], "plan": [], "reflection_round": 0, "gaps": []},
                 config={"recursion_limit": settings.max_steps * 3 + 20})
    seconds = round(time.monotonic() - started, 1)
    metrics = {"run_id": run_dir.name, "variant": variant, "as_of": as_of.isoformat(), "seconds": seconds,
               "model": client.model_name, "ocr": f"{ocr_engine.name}/{settings.ocr_preprocess}",
               "prompt_versions": prompt_versions(), "llm_usage": dict(client.usage),
               "cache_hits": getattr(client, "hits", 0), "documents": len(ctx.documents),
               "unprocessed": len(ctx.unprocessed), "removed_claims": ctx.removed_claims, "safety_net": ctx.safety_net}
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    return RunResult(run_dir=run_dir, report_path=ctx.report_path, kb_dir=run_dir / "kb", trace_path=trace.path,
                     variant=variant, seconds=seconds, llm_usage=dict(client.usage))
