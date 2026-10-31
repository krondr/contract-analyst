"""Run experiment variants, score them on the golden set and write a metrics-only markdown summary."""

from __future__ import annotations

import json
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


def run_variant(spec: VariantSpec, data_dir: Path, golden: GoldenSet, as_of: date,
                out_dir: Path = Path("runs/experiments"), **run_kwargs) -> VariantResult:
    settings = load_settings(**spec.overrides)
    if spec.variant == "S":
        from contract_analyst.agent.single_call import run_single_call

        result = run_single_call(data_dir, settings, as_of, out_dir, **run_kwargs)
    else:
        from contract_analyst.agent.runner import run_analysis

        result = run_analysis(data_dir, settings, as_of, spec.variant, out_dir=out_dir, console=False, **run_kwargs)
    return VariantResult(spec=spec, scores=score_run(result.run_dir, golden), run_dir=result.run_dir)


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
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def experiment_specs(name: str, models: list[str] | None = None, variants: list[str] | None = None) -> list[VariantSpec]:
    if name == "e2":
        return [VariantSpec(name=m, overrides={"llm_model": m}) for m in (models or ["ollama:qwen3:8b"])]
    if name == "e3":
        return [VariantSpec(name=mode, overrides={"consolidation_mode": mode}) for mode in ("code", "llm")]
    if name == "e4":
        return [VariantSpec(name=v, variant=v) for v in (variants or ["A", "B", "S"])]
    raise ValueError(f"unknown experiment {name}")


def decide(name: str, results: list[VariantResult]) -> tuple[str, str, str]:
    """Title, pre-registered rule and winner for an experiment. Rules arrive with E2–E4."""
    raise NotImplementedError(f"selection rule for {name} is defined in its experiment task")
