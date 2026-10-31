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
            out: Annotated[Path, typer.Option(help="Папка для результатов")] = Path("runs")) -> None:
    """Разобрать пакет документов и собрать отчёт."""
    from datetime import date

    from contract_analyst.agent.runner import run_analysis
    from contract_analyst.llm.client import LLMUnavailable, ReplayMiss

    overrides = {k: v for k, v in {"llm_model": model, "llm_cache_mode": llm_cache, "agent_variant": variant}.items() if v}
    settings = load_settings(**overrides)
    when = date.fromisoformat(as_of) if as_of else date.today()
    try:
        result = run_analysis(data, settings, when, settings.agent_variant, only_chain=chain, only_doc=doc, out_dir=out)
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
    """Автопроверка эталона: цитаты, форматы, пересчёт свёртки, ловушки."""
    from contract_analyst.eval.golden import load_golden
    from contract_analyst.eval.golden_tools import check_golden

    docs = {d.doc_id: d for d in _read_all(data)}
    traps = {t.strip() for t in required_traps.split(",") if t.strip()} or None
    errors = check_golden(load_golden(golden), docs, traps)
    for error in errors:
        typer.echo(error)
    typer.echo(f"{len(errors)} errors")
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
