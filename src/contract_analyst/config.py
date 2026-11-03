"""Runtime settings: YAML defaults, then environment, then explicit overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "default.yaml"


def disable_remote_tracing() -> None:
    """Documents never leave the machine. LangChain and LangGraph send every run to LangSmith by themselves when the
    environment enables tracing and holds an API key; force it off for this process. Every entry point imports
    this module before LangChain, and LangSmith reads (and caches) these variables only when the first run starts."""
    for flag in ("LANGSMITH_TRACING_V2", "LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING", "LANGCHAIN_TRACING"):
        os.environ[flag] = "false"
    for key in ("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY"):
        os.environ.pop(key, None)


disable_remote_tracing()


class Settings(BaseModel):
    llm_model: str = "ollama:qwen3:8b"
    ollama_url: str = "http://localhost:11434"
    openai_base_url: str = "http://localhost:8000/v1"  # for openai:<model>
    openai_extra_body: dict = {}  # server-specific request fields for extraction calls
    openai_think_extra_body: dict = {}  # ... and for the agent's thinking turns
    llm_num_ctx: int = 16384
    llm_batch_tokens: int = 6000
    llm_max_tokens: int = 4096
    llm_think_max_tokens: int = 4096
    llm_keep_alive: str = "30m"
    llm_call_deadline: float = 900
    llm_cache_mode: Literal["record", "replay", "off"] = "record"
    ocr_engine: Literal["tesseract", "rapidocr"] = "rapidocr"
    ocr_preprocess: Literal["none", "deskew_sauvola"] = "none"
    ocr_dpi: int = 300
    ocr_word_conf_threshold: float = 0.90
    ocr_quote_fuzzy_threshold: int = 90
    text_layer_min_chars: int = 50
    low_quality_ratio: float = 0.2
    agent_variant: Literal["A", "B", "C"] = "B"
    consolidation_mode: Literal["code", "llm"] = "code"
    max_steps: int = 60
    max_seconds: int = 7200
    max_tool_errors_in_row: int = 5
    max_reflection_rounds: int = 2
    agent_thinking: bool = True
    subject_categories: dict[str, str] = {
        "commercial_representation": "Коммерческое представительство",
        "partner_promotion": "Партнёрское продвижение",
        "sales_targets_agreement": "Соглашение о плановых показателях",
        "lease": "Аренда",
        "sale": "Купля-продажа",
        "services": "Оказание услуг",
        "supply": "Поставка",
        "other": "Иное",
    }
    field_groups: dict[str, list[str]] = {
        "header": ["kind", "number", "doc_date", "city", "parent", "effective_from", "effective_to",
                   "parties", "subject_category"],
        "core_terms": ["subject", "validity", "payment_terms", "termination", "notification_channel",
                       "penalties"],
        "money": ["rates"],
        "dealer": ["plan_indicators", "tariff_groups", "trade_points"],
        "ops": ["ops"],
    }
    field_descriptions: dict[str, str] = {
        "kind": ("Вид документа: основной договор, дополнительное соглашение, рамочное соглашение "
                 "или неизвестно"),
        "number": "Номер документа из заголовка",
        "doc_date": "Дата подписания документа",
        "city": "Город заключения документа",
        "parent": "Номер и дата договора, к которому относится соглашение, как они написаны в тексте",
        "effective_from": "Дата вступления документа в силу или дата, с которой он распространяет действие",
        "effective_to": ("Дата, до которой действует сам документ, если она указана "
                         "(например, соглашение на один месяц)"),
        "parties": "Стороны: роль, организационно-правовая форма, наименование",
        "subject_category": "Категория предмета договора из закрытого списка",
        "subject": "Предмет договора: что одна сторона поручает или передаёт другой",
        "validity": "Срок действия договора: дата окончания или порядок пролонгации",
        "payment_terms": "Порядок и сроки расчётов и выплат",
        "termination": "Порядок и основания расторжения",
        "notification_channel": "Способ и адреса для уведомлений и обмена документами",
        "penalties": "Штрафы и неустойки: за что и в каком размере",
        "rates": "Ставки цены или вознаграждения: название, сумма, НДС, период действия",
        "plan_indicators": "Плановые показатели продаж или начислений",
        "tariff_groups": "Группы тарифных планов и их состав",
        "trade_points": "Торговые точки, на которых исполняется договор",
        "ops": "Операции изменения: какой пункт, раздел, приложение или документ меняется и с какой даты",
    }
    critical_fields: list[str] = ["number", "doc_date", "parent", "effective_from", "rates", "ops"]


def load_settings(path: Path | None = None, **overrides) -> Settings:
    """Defaults, then a deployment file named by CA_CONFIG (kept outside the repo), then the URL variables, then
    explicit overrides."""
    data = yaml.safe_load((path or DEFAULT_CONFIG).read_text(encoding="utf-8")) or {}
    if extra := os.environ.get("CA_CONFIG"):
        data.update(yaml.safe_load(Path(extra).expanduser().read_text(encoding="utf-8")) or {})
    if url := os.environ.get("CA_OLLAMA_URL"):
        data["ollama_url"] = url
    if url := os.environ.get("CA_OPENAI_BASE_URL"):
        data["openai_base_url"] = url
    data.update(overrides)
    return Settings.model_validate(data)
