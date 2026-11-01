"""One analysis run: build the context, run the chosen orchestration variant, write metrics."""

from __future__ import annotations

import json
import time
from datetime import date, datetime
from pathlib import Path
from typing import Literal

from langgraph.errors import GraphRecursionError
from pydantic import BaseModel

from contract_analyst.agent.context import RunContext
from contract_analyst.agent.trace import TraceLogger, read_trace, thought_coverage
from contract_analyst.config import Settings
from contract_analyst.ingest.inventory import input_paths, path_key
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


class InputError(ValueError):
    """The input leaves nothing to analyze: a missing or empty folder, or a selection that matches no file."""


def _selected_paths(input_dir: Path, only_doc: Path | None) -> set[str] | None:
    if not input_dir.is_dir():
        raise InputError(f"Папка {input_dir} не найдена")
    paths = input_paths(input_dir)
    if not paths:
        raise InputError(f"В папке {input_dir} нет файлов для разбора")
    if only_doc is None:
        return None
    # The file as given, or relative to the input folder; compared resolved and in NFC.
    found = {path_key(p) for p in paths} & {path_key(only_doc), path_key(input_dir / only_doc)}
    if not found:
        names = ", ".join(p.relative_to(input_dir).as_posix() for p in paths[:10])
        raise InputError(f"Файла {only_doc} нет в папке {input_dir}. Файлы в папке: {names}"
                         + (" …" if len(paths) > 10 else ""))
    return found


def _restrict_to_chain(ctx: RunContext, number: str) -> None:
    ctx.only_doc_ids = {doc_id for doc_id, h in ctx.headers.items() if number in (h.number, h.parent_number)}
    if not ctx.only_doc_ids:
        raise InputError(f"Цепочки {number} нет: ни один документ не называет этот номер в заголовке")


def run_analysis(input_dir: Path, settings: Settings, as_of: date, variant: Literal["A", "B", "C"] = "B",
                 only_chain: str | None = None, only_doc: Path | None = None, out_dir: Path = Path("runs"),
                 client: LLMClient | None = None, ocr_engine: OcrEngine | None = None,
                 read_cache: Path | None = DEFAULT_READ_CACHE, llm_cache: Path = DEFAULT_LLM_CACHE,
                 console: bool = True) -> RunResult:
    only_paths = _selected_paths(input_dir, only_doc)
    if only_chain and variant == "C":
        raise InputError("--chain не поддерживается в варианте C: там агент сам читает файлы")
    run_dir = _run_dir(out_dir, variant)
    if client is None:
        client = get_llm_client(settings, llm_cache)
        if hasattr(client, "check_available"):
            client.check_available()
    ocr_engine = ocr_engine or get_ocr_engine(settings.ocr_engine)
    trace = TraceLogger(run_dir, run_dir.name, console=console)
    ctx = RunContext(settings=settings, client=client, ocr_engine=ocr_engine, trace=trace, input_dir=input_dir,
                     run_dir=run_dir, as_of=as_of, read_cache=read_cache,
                     only_paths=only_paths)
    started = time.monotonic()
    if variant == "B":
        from contract_analyst.agent.graph_hybrid import build_hybrid_graph

        graph = build_hybrid_graph(ctx)
        if only_chain:
            from contract_analyst.agent import steps

            steps.ingest_all(ctx)
            _restrict_to_chain(ctx, only_chain)
            graph = build_hybrid_graph(ctx, with_ingest=False)
    elif variant == "A":
        from contract_analyst.agent.graph_pipeline import build_pipeline_graph

        if only_chain:
            from contract_analyst.agent import steps

            steps.ingest_all(ctx)
            _restrict_to_chain(ctx, only_chain)
        graph = build_pipeline_graph(ctx)
    else:
        from contract_analyst.agent.graph_full import build_full_agent_graph

        graph = build_full_agent_graph(ctx)
    try:
        graph.invoke({"messages": [], "plan": [], "reflection_round": 0, "gaps": []},
                     config={"recursion_limit": settings.max_steps * 3 + 20})
    except GraphRecursionError as exc:  # the graph was cut off: still finish the work and write the report
        from contract_analyst.agent.graph_hybrid import report

        ctx.trace.event("error", "runner", message=f"GraphRecursionError: {exc}"[:300])
        report(ctx)
    seconds = round(time.monotonic() - started, 1)
    metrics = {"run_id": run_dir.name, "variant": variant, "as_of": as_of.isoformat(), "seconds": seconds,
               "model": client.model_name, "ocr": f"{ocr_engine.name}/{settings.ocr_preprocess}",
               "prompt_versions": prompt_versions(), "llm_usage": dict(client.usage),
               "cache_hits": getattr(client, "hits", 0), "documents": len(ctx.documents),
               "unprocessed": len(ctx.unprocessed), "removed_claims": ctx.removed_claims, "safety_net": ctx.safety_net,
               "scope": {"chain": only_chain, "doc": str(only_doc) if only_doc else None},
               "thought_coverage": thought_coverage(read_trace(trace.path))}
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    return RunResult(run_dir=run_dir, report_path=ctx.report_path, kb_dir=run_dir / "kb", trace_path=trace.path,
                     variant=variant, seconds=seconds, llm_usage=dict(client.usage))
