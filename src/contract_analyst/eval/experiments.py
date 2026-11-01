"""Run experiment variants, score them on the golden set and write a metrics-only markdown summary."""

from __future__ import annotations

import json
import re
import subprocess
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from contract_analyst.config import load_settings
from contract_analyst.eval.golden import GoldenSet
from contract_analyst.eval.metrics import Scores, score_run


class VariantSpec(BaseModel):
    name: str
    overrides: dict = {}
    variant: Literal["A", "B", "C", "S"] = "B"


class VariantResult(BaseModel):
    spec: VariantSpec
    scores: Scores
    run_dir: Path
    runs_at: dict[date, Path] = {}


def _run_once(spec: VariantSpec, settings, data_dir: Path, as_of: date, out_dir: Path, run_kwargs: dict) -> Path:
    if spec.variant == "S":
        from contract_analyst.agent.single_call import run_single_call

        return run_single_call(data_dir, settings, as_of, out_dir, **run_kwargs).run_dir
    from contract_analyst.agent.runner import run_analysis

    return run_analysis(data_dir, settings, as_of, spec.variant, out_dir=out_dir, console=False, **run_kwargs).run_dir


def run_variant(spec: VariantSpec, data_dir: Path, golden: GoldenSet, as_of: date,
                out_dir: Path = Path("runs/experiments"), **run_kwargs) -> VariantResult:
    """Run the variant once per golden date and score each date from its own run: an LLM or single-call fold made
    at one date says nothing about another. Extraction is cached, so the extra runs are cheap. Everything that does
    not depend on the date (values, chains, clusters, time, calls) comes from the run at as_of."""
    settings = load_settings(**spec.overrides)
    runs = {when: _run_once(spec, settings, data_dir, when, out_dir, run_kwargs)
            for when in dict.fromkeys([as_of, *golden.as_of_dates])}
    runs_at = {when: runs[when] for when in golden.as_of_dates}
    return VariantResult(spec=spec, scores=score_run(runs[as_of], golden, runs_at), run_dir=runs[as_of],
                         runs_at=runs_at)


COLUMNS = [("Неверных принятых", "critical_accepted_wrong", "{}"), ("Доля принятых верных", "critical_coverage", "{:.2f}"),
           ("Свёртка", "consolidation_acc", "{:.2f}"), ("Ловушки", None, None), ("Цепочки", "chain_acc", "{:.2f}"),
           ("Вид документа", "doc_kind_acc", "{:.2f}"), ("Валидный JSON", "json_valid_rate", "{:.2f}"),
           ("Валидные вызовы", "tool_call_valid_rate", "{:.2f}"), ("Время, с", "wall_seconds", "{:.0f}"),
           ("Вызовов LLM", "llm_calls", "{}"), ("Подстраховка кодом", None, None)]


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _row(r: VariantResult) -> list[str]:
    s, cells = r.scores, [r.spec.name]
    for _, attr, fmt in COLUMNS:
        if attr:
            cells.append(fmt.format(getattr(s, attr)))
        elif _ == "Ловушки":
            cells.append(f"{s.traps_passed}/{s.traps_total}")
        else:
            cells.append(", ".join(s.safety_net) or "—")
    return cells


def save_results(name: str, results: list[VariantResult], out_dir: Path, append: bool = False) -> list[VariantResult]:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = out_dir / f"{name}.json"
    previous = [VariantResult.model_validate(x) for x in json.loads(raw.read_text(encoding="utf-8"))] if append and raw.exists() else []
    merged = [p for p in previous if p.spec.name not in {r.spec.name for r in results}] + results
    raw.write_text(json.dumps([r.model_dump(mode="json") for r in merged], ensure_ascii=False, indent=1), encoding="utf-8")
    return merged


def write_experiment_md(name: str, title: str, results: list[VariantResult], rule: str, winner: str, out: Path,
                        notes: str = "") -> Path:
    meta = {}
    if results and (results[0].run_dir / "metrics.json").exists():
        meta = json.loads((results[0].run_dir / "metrics.json").read_text(encoding="utf-8"))
    header = "| Вариант | " + " | ".join(c for c, _, _ in COLUMNS) + " |"
    lines = [f"# {name.upper()}: {title}", "", "Метрики на эталоне реальных документов (dev). Тексты договоров не приводятся.",
             "", header, "|" + "---|" * (len(COLUMNS) + 1)]
    lines += ["| " + " | ".join(_row(r)) + " |" for r in results]
    lines += ["", f"**Правило выбора (записано до замера):** {rule}", "", f"**Победитель:** {winner}", ""]
    if notes:
        lines += [notes, ""]
    lines += ["Метаданные:", f"- git: `{_git_sha()}`", f"- модель первого прогона: `{meta.get('model', '—')}`",
              f"- OCR: `{meta.get('ocr', '—')}`", f"- версии промптов: `{meta.get('prompt_versions', {})}`",
              f"- дата свёртки: `{meta.get('as_of', '—')}`"]
    if dates := sorted({when for r in results for when in r.runs_at}):
        lines.append(f"- даты эталона: {', '.join(map(str, dates))}, каждая оценена по своему прогону")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def experiment_specs(name: str, models: list[str] | None = None, variants: list[str] | None = None,
                     model: str | None = None) -> list[VariantSpec]:
    """`model` runs e3 and e4 on that model instead of the configured one."""
    on = {"llm_model": model} if model else {}
    if name == "e2":
        return [VariantSpec(name=m, overrides={"llm_model": m}) for m in (models or ["ollama:qwen3:8b"])]
    if name == "e3":
        return [VariantSpec(name=mode, overrides={"consolidation_mode": mode, **on}) for mode in ("code", "llm")]
    if name == "e4":
        return [VariantSpec(name=v, variant=v, overrides=on) for v in (variants or ["A", "B", "S"])]
    raise ValueError(f"unknown experiment {name}")


_SIZE = re.compile(r"(\d+(?:\.\d+)?)b", re.IGNORECASE)

RULES = {
    "e2": ("Выбор LLM", ("1) отбрасываются модели с неверными принятыми критичными значениями, если есть модель без них; "
           "2) максимум доли принятых верных (ничья — разница ≤ 0,02); 3) доля валидных JSON и вызовов инструментов; "
           "4) меньше время; 5) при ничьей — модель меньше.")),
    "e3": ("Свёртка кодом или LLM", ("побеждает code, если LLM не лучше по точности свёртки больше чем на один договор "
           "(0,2 при пяти договорах); при ничьей — code (прослеживаемость).")),
    "e4": ("Оркестрация", ("среди агентных вариантов B и C — больше пройденных ловушек, затем выше точность свёртки; "
           "при ничьей — B. Конвейер A и один вызов S — точки отсчёта, в финал не идут.")),
}


def _model_size(name: str) -> float:
    m = _SIZE.search(name)
    return float(m.group(1)) if m else float("inf")


def select_e2_winner(results: list[VariantResult]) -> VariantResult:
    pool = [r for r in results if r.scores.critical_accepted_wrong == 0] or list(results)
    best = max(r.scores.critical_coverage for r in pool)
    pool = [r for r in pool if best - r.scores.critical_coverage <= 0.02]
    validity = max(r.scores.json_valid_rate + r.scores.tool_call_valid_rate for r in pool)
    pool = [r for r in pool if validity - (r.scores.json_valid_rate + r.scores.tool_call_valid_rate) <= 0.01]
    return min(pool, key=lambda r: (r.scores.wall_seconds, _model_size(r.spec.name)))


def select_e3_winner(results: list[VariantResult], contracts: int = 5) -> VariantResult:
    by_name = {r.spec.name: r for r in results}
    code, llm = by_name.get("code"), by_name.get("llm")
    if code is None or llm is None:
        return code or llm
    margin = 1 / contracts
    return llm if llm.scores.consolidation_acc - code.scores.consolidation_acc > margin else code


def select_e4_winner(results: list[VariantResult]) -> VariantResult:
    agents = [r for r in results if r.spec.variant in ("B", "C")]
    return max(agents, key=lambda r: (r.scores.traps_passed, round(r.scores.consolidation_acc, 4),
                                      r.spec.variant == "B"))


def decide(name: str, results: list[VariantResult]) -> tuple[str, str, str]:
    """Title, pre-registered rule and winner for an experiment."""
    title, rule = RULES[name]
    select = {"e2": select_e2_winner, "e3": select_e3_winner, "e4": select_e4_winner}[name]
    return title, rule, select(results).spec.name
