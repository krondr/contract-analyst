"""Runtime settings: YAML defaults, then environment, then explicit overrides."""

from __future__ import annotations

import os
from importlib.resources import files
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[2]


def resource_path(relative: str) -> Path:
    """A file or folder of the repo, such as config/default.yaml or prompts. An installed wheel has no repo around
    the module and carries copies inside the package instead (force-include in pyproject.toml)."""
    in_repo = REPO_ROOT / relative
    return in_repo if in_repo.exists() else Path(files("contract_analyst"), "_resources", relative)


DEFAULT_CONFIG = resource_path("config/default.yaml")


def disable_remote_tracing() -> None:
    """Documents never leave the machine. LangChain and LangGraph send every run to LangSmith by themselves when the
    environment enables tracing and holds an API key; force it off for this process. Every entry point imports
    this module before LangChain, and LangSmith reads (and caches) these variables only when the first run starts."""
    for flag in ("LANGSMITH_TRACING_V2", "LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING", "LANGCHAIN_TRACING"):
        os.environ[flag] = "false"
    for key in ("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY"):
        os.environ.pop(key, None)


disable_remote_tracing()



def _yaml_defaults() -> dict:
    return yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8")) or {}


class Settings(BaseModel):
    llm_model: str = "ollama:qwen3:8b"
    ollama_url: str = "http://localhost:11434"
    openai_base_url: str = "http://localhost:8000/v1"  # for openai:<model>
    openai_extra_body: dict = {}  # server-specific request fields for extraction calls
    openai_think_extra_body: dict = {}  # ... and for the agent's thinking turns
    llm_num_ctx: int = 16384
    llm_batch_tokens: int = 6000
    llm_money_batch_tokens: int = 6000
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
    # These, like every other key of config/default.yaml, take their defaults from that file (see __init__).
    subject_categories: dict[str, str] = {}
    field_groups: dict[str, list[str]] = {}
    field_descriptions: dict[str, str] = {}
    critical_fields: list[str] = []

    def __init__(self, **data) -> None:
        """config/default.yaml is the one source of defaults: a value changed there is what Settings() gives;
        explicit values win. load_settings() validates the merged layers directly and does not come through here."""
        super().__init__(**{**_yaml_defaults(), **data})


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
