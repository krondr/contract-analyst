"""Build the LLM client from settings: `ollama:<model>` (cached) or `scripted:<path.json>`."""

from __future__ import annotations

from pathlib import Path

from contract_analyst.config import Settings
from contract_analyst.llm.cache import CachingClient
from contract_analyst.llm.client import LLMClient
from contract_analyst.llm.ollama_client import OllamaClient
from contract_analyst.llm.scripted import ScriptedClient, load_script


def get_llm_client(settings: Settings, cache_db: Path) -> LLMClient:
    spec = settings.llm_model
    if spec.startswith("ollama:"):
        inner = OllamaClient(spec.removeprefix("ollama:"), settings.ollama_url, settings.llm_num_ctx,
                             max_tokens=settings.llm_max_tokens, think_max_tokens=settings.llm_think_max_tokens,
                             keep_alive=settings.llm_keep_alive, deadline=settings.llm_call_deadline)
        return CachingClient(inner, cache_db, settings.llm_cache_mode)
    if spec.startswith("scripted:"):
        return ScriptedClient(load_script(Path(spec.removeprefix("scripted:"))))
    raise ValueError(f"unknown LLM spec: {spec!r} (expected ollama:<model> or scripted:<file>)")
