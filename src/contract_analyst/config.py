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
    llm_num_ctx: int = 16384
    llm_batch_tokens: int = 6000
    llm_max_tokens: int = 4096
    llm_think_max_tokens: int = 4096
    llm_keep_alive: str = "30m"
    llm_call_deadline: float = 900
    llm_cache_mode: Literal["record", "replay", "off"] = "record"
    ocr_engine: Literal["tesseract", "rapidocr"] = "tesseract"
    ocr_preprocess: Literal["none", "deskew_sauvola"] = "none"
    ocr_dpi: int = 300
    ocr_word_conf_threshold: float = 0.80
    ocr_quote_fuzzy_threshold: int = 90
    text_layer_min_chars: int = 50
    low_quality_ratio: float = 0.2
    agent_variant: Literal["A", "B", "C"] = "B"
    consolidation_mode: Literal["code", "llm"] = "code"
    max_steps: int = 60
    max_seconds: int = 1800
    max_tool_errors_in_row: int = 5
    max_reflection_rounds: int = 2
    agent_thinking: bool = True
    subject_categories: dict[str, str] = {}
    field_groups: dict[str, list[str]] = {}
    field_descriptions: dict[str, str] = {}
    critical_fields: list[str] = []


def load_settings(path: Path | None = None, **overrides) -> Settings:
    data = yaml.safe_load((path or DEFAULT_CONFIG).read_text(encoding="utf-8")) or {}
    if url := os.environ.get("CA_OLLAMA_URL"):
        data["ollama_url"] = url
    data.update(overrides)
    return Settings.model_validate(data)
