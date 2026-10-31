"""Command line interface: analyze, chat, eval, experiments, OCR benchmark and golden set tools."""

from __future__ import annotations

from pathlib import Path

import typer

from contract_analyst.config import load_settings

app = typer.Typer(no_args_is_help=True, help="Аналитик цепочек и кластеров договоров")
golden_app = typer.Typer(no_args_is_help=True, help="Инструменты эталона")
app.add_typer(golden_app, name="golden")


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
