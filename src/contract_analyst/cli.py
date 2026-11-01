"""Command line interface: analyze, chat, eval, experiments, OCR benchmark and golden set tools."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from contract_analyst.config import load_settings

app = typer.Typer(no_args_is_help=True, help="Аналитик цепочек и кластеров договоров")
golden_app = typer.Typer(no_args_is_help=True, help="Инструменты эталона")
eval_app = typer.Typer(no_args_is_help=True, help="Оценка прогонов")
app.add_typer(golden_app, name="golden")
app.add_typer(eval_app, name="eval")


@app.command()
def analyze(data: Path,
            as_of: Annotated[str | None, typer.Option(help="Дата свёртки YYYY-MM-DD, по умолчанию сегодня")] = None,
            variant: Annotated[str | None, typer.Option(help="Оркестрация: A, B или C")] = None,
            model: Annotated[str | None, typer.Option(help="Модель, например ollama:qwen3:8b")] = None,
            chain: Annotated[str | None, typer.Option(help="Номер договора: разобрать только эту цепочку")] = None,
            doc: Annotated[Path | None, typer.Option(help="Разобрать только этот файл")] = None,
            llm_cache: Annotated[str | None, typer.Option(help="record | replay | off")] = None,
            cache_file: Annotated[Path, typer.Option(help="Файл кэша ответов модели")] = Path("data/cache/llm.sqlite"),
            out: Annotated[Path, typer.Option(help="Папка для результатов")] = Path("runs")) -> None:
    """Разобрать пакет документов и собрать отчёт."""
    from datetime import date

    from contract_analyst.agent.runner import run_analysis
    from contract_analyst.llm.client import LLMUnavailable, ReplayMiss

    overrides = {k: v for k, v in {"llm_model": model, "llm_cache_mode": llm_cache, "agent_variant": variant}.items() if v}
    settings = load_settings(**overrides)
    when = date.fromisoformat(as_of) if as_of else date.today()
    try:
        result = run_analysis(data, settings, when, settings.agent_variant, only_chain=chain, only_doc=doc, out_dir=out,
                              llm_cache=cache_file)
    except LLMUnavailable as exc:
        typer.echo(exc.hint)
        raise typer.Exit(2) from None
    except ReplayMiss as exc:
        typer.echo(f"Нет записанного ответа модели для этого запуска (режим replay): {exc}")
        raise typer.Exit(2) from None
    typer.echo(f"Отчёт: {result.report_path}\nБаза знаний: {result.kb_dir}\nТрасса: {result.trace_path}\n"
               f"Время: {result.seconds} с; вызовов LLM: {result.llm_usage.get('calls', 0)}")


def _read_all(data: Path, cache: Path | None = None):
    from contract_analyst.ingest.inventory import list_inputs
    from contract_analyst.ingest.reader import read_document
    from contract_analyst.ocr import get_ocr_engine

    settings = load_settings()
    engine = get_ocr_engine(settings.ocr_engine)
    files = [f for f in list_inputs(data) if f.duplicate_of is None]
    return [read_document(f, engine, settings, cache_dir=cache or Path("data/cache/read")) for f in files]


@golden_app.command("prefill")
def golden_prefill(data: Path, out: Path = Path("data/golden/prefill.yaml")) -> None:
    """Предзаполнить эталон номерами, датами и ссылками из заголовков."""
    from contract_analyst.eval.golden import dump_golden
    from contract_analyst.eval.golden_tools import prefill

    dump_golden(prefill(_read_all(data), root=data), out)
    typer.echo(f"written {out}")


@golden_app.command("dump-texts")
def golden_dump_texts(data: Path, out: Path = Path("data/golden/texts")) -> None:
    """Выгрузить тексты документов (и PNG страниц-сканов) для разметчиков."""
    from contract_analyst.eval.golden_tools import dump_texts

    paths = dump_texts(_read_all(data), out)
    typer.echo(f"written {len(paths)} files to {out}")


@golden_app.command("diff")
def golden_diff(a: Path, b: Path, out: Path = Path("data/golden/disputed.json")) -> None:
    """Сравнить две независимые разметки и записать расхождения."""
    from contract_analyst.eval.golden import load_golden
    from contract_analyst.eval.golden_tools import diff_golden, diffs_to_json

    diffs = diff_golden(load_golden(a), load_golden(b))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(diffs_to_json(diffs), encoding="utf-8")
    typer.echo(f"{len(diffs)} disagreements written to {out}")


@golden_app.command("check")
def golden_check(golden: Path, data: Path = Path("data/raw"), required_traps: str = "") -> None:
    """Автопроверка эталона: цитаты, форматы, ссылки, ловушки; расхождения со свёрткой кодом — на разбор."""
    from contract_analyst.eval.golden import load_golden
    from contract_analyst.eval.golden_tools import check_golden, fold_mismatches

    docs = {d.doc_id: d for d in _read_all(data)}
    traps = {t.strip() for t in required_traps.split(",") if t.strip()} or None
    gs = load_golden(golden)
    errors, disputed = check_golden(gs, docs, traps), fold_mismatches(gs, docs)
    for error in errors:
        typer.echo(error)
    for item in disputed:
        typer.echo(f"needs adjudication: {item}")
    typer.echo(f"{len(errors)} errors, {len(disputed)} differences from the code fold need adjudication")
    raise typer.Exit(1 if errors else 0)


@golden_app.command("review")
def golden_review(golden: Path, disputed: Path = Path("data/golden/disputed.json"),
                  out: Path = Path("data/golden/review.html")) -> None:
    """Собрать страницу для проверки эталона человеком."""
    import json

    from contract_analyst.eval.golden import load_golden
    from contract_analyst.eval.golden_tools import GoldenDiff, build_review_page

    diffs = [GoldenDiff(**d) for d in json.loads(disputed.read_text(encoding="utf-8"))] if disputed.exists() else []
    typer.echo(f"written {build_review_page(load_golden(golden), diffs, out)}")


@eval_app.command("golden")
def eval_golden(run: Path, golden: Path = Path("data/golden/golden.yaml")) -> None:
    """Сравнить прогон с эталоном: метрики и ловушки."""
    from contract_analyst.eval.golden import load_golden
    from contract_analyst.eval.metrics import score_run
    from contract_analyst.eval.traps import check_traps

    gs = load_golden(golden)
    typer.echo(score_run(run, gs).model_dump_json(indent=1))
    for trap in check_traps(run, gs):
        typer.echo(f"{'OK ' if trap.passed else 'FAIL'} {trap.trap_id}: {trap.detail}")


@app.command()
def experiment(name: str, data: Path = Path("data/raw"), golden: Path = Path("data/golden/golden.yaml"),
               as_of: str = "2026-10-30", models: str = "", variants: str = "A,B,S",
               append: bool = False, out: Path = Path("docs/experiments")) -> None:
    """Замеры: e2 (модели), e3 (свёртка кодом или LLM), e4 (оркестрация)."""
    from datetime import date

    from contract_analyst.eval import experiments as ex
    from contract_analyst.eval.golden import load_golden

    gs = load_golden(golden)
    specs = ex.experiment_specs(name, models=[m for m in models.split(",") if m], variants=variants.split(","))
    results = [ex.run_variant(spec, data, gs, date.fromisoformat(as_of)) for spec in specs]
    merged = ex.save_results(name, results, out, append=append)
    title, rule, winner = ex.decide(name, merged)
    typer.echo(f"written {ex.write_experiment_md(name, title, merged, rule, winner, out / f'{name}.md')}")


@app.command()
def chat(kb: Path, thread: str = "default", model: str | None = None) -> None:
    """Вопросы по базе знаний прогона, с памятью диалога (пустая строка — выход)."""
    from contract_analyst.chat.graph import ChatSession
    from contract_analyst.kb.store import KnowledgeBase
    from contract_analyst.llm.client import LLMUnavailable
    from contract_analyst.llm.factory import get_llm_client

    settings = load_settings(**({"llm_model": model} if model else {}))
    client = get_llm_client(settings, Path("data/cache/llm.sqlite"))
    try:
        if hasattr(client, "check_available"):
            client.check_available()
    except LLMUnavailable as exc:
        typer.echo(exc.hint)
        raise typer.Exit(2) from None
    session = ChatSession(KnowledgeBase(kb), client, settings, kb / "chat.sqlite")
    while question := typer.prompt("Вопрос", default="", show_default=False).strip():
        typer.echo(session.ask(question, thread))


@eval_app.command("chat")
def eval_chat(kb: Path, questions: Path = Path("data/golden/chat_questions.yaml"), model: str | None = None) -> None:
    """Оценить ответы чата: факты, отказы, ссылки."""
    import tempfile

    from contract_analyst.eval.chat_eval import chat_eval
    from contract_analyst.llm.factory import get_llm_client

    settings = load_settings(**({"llm_model": model} if model else {}))
    with tempfile.TemporaryDirectory() as tmp:
        scores = chat_eval(kb, questions, get_llm_client(settings, Path("data/cache/llm.sqlite")), settings,
                           Path(tmp) / "memory.sqlite")
    typer.echo(scores.model_dump_json(indent=1))


# --- E1: OCR benchmark -------------------------------------------------------------------------------------------


@app.command("bench-ocr")
def bench_ocr(
    data: Path,
    out: Annotated[Path, typer.Option(help="Куда записать результаты")] = Path("docs/experiments/e1_ocr.md"),
    golden: Annotated[Path | None, typer.Option(
        help="Эталон с расшифровками сканов; по умолчанию <data>/../golden/golden.yaml")] = None,
    pages: Annotated[int, typer.Option(help="Сколько синтетических сканов собрать")] = 40,
    seed: Annotated[int, typer.Option(help="Seed выбора страниц и искажений")] = 0,
    engines: Annotated[str, typer.Option(help="Движки через запятую")] = "tesseract,rapidocr",
    modes: Annotated[str, typer.Option(help="Варианты предобработки через запятую")] = "none,deskew_sauvola",
    note: Annotated[str, typer.Option(help="Причина прогона; попадёт в документ")] = "",
) -> None:
    """E1: замер OCR-движков и предобработки на синтетических сканах и расшифровках из эталона (только метрики)."""
    from contract_analyst.eval.ocr_bench import run_benchmark

    rows, winner = run_benchmark(data, out, golden=golden, n_pages=pages, seed=seed,
                                 engines=[e for e in engines.split(",") if e],
                                 modes=[m for m in modes.split(",") if m], note=note,
                                 log=lambda message: typer.echo(message, err=True))
    for r in rows:
        threshold = "—" if r.threshold is None else f"{r.threshold:.2f}"
        typer.echo(f"{r.engine} + {r.mode}: CER {r.cer:.4f}, покрытие при точности ≥ 0.99 "
                   f"{r.crit_coverage_at_p99:.3f} (t = {threshold}), {r.sec_per_page:.2f} с/стр")
    rec = winner.recommended_threshold
    typer.echo(f"победитель: {winner.engine} + {winner.mode}; "
               f"ocr_word_conf_threshold: {'—' if rec is None else f'{rec:.2f}'}")
    typer.echo(f"записано: {out}")
