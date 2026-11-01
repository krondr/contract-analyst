"""Build the LLM client from settings: `ollama:<model>`, `openai:<model>` (both cached) or `scripted:<path.json>`."""

from __future__ import annotations

import os
from pathlib import Path

from contract_analyst.config import Settings
from contract_analyst.llm.cache import CachingClient
from contract_analyst.llm.client import LLMClient
from contract_analyst.llm.ollama_client import OllamaClient
from contract_analyst.llm.openai_client import OpenAICompatClient
from contract_analyst.llm.scripted import ScriptedClient, load_script


def get_llm_client(settings: Settings, cache_db: Path) -> LLMClient:
    spec = settings.llm_model
    if spec.startswith("ollama:"):
        inner = OllamaClient(spec.removeprefix("ollama:"), settings.ollama_url, settings.llm_num_ctx,
                             max_tokens=settings.llm_max_tokens, think_max_tokens=settings.llm_think_max_tokens,
                             keep_alive=settings.llm_keep_alive, deadline=settings.llm_call_deadline)
        return CachingClient(inner, cache_db, settings.llm_cache_mode)
    if spec.startswith("openai:"):
        inner = OpenAICompatClient(spec.removeprefix("openai:"), settings.openai_base_url,
                                   os.environ.get("CA_OPENAI_API_KEY"), max_tokens=settings.llm_max_tokens,
                                   think_max_tokens=settings.llm_think_max_tokens, deadline=settings.llm_call_deadline,
                                   extra_body=settings.openai_extra_body,
                                   think_extra_body=settings.openai_think_extra_body)
        return CachingClient(inner, cache_db, settings.llm_cache_mode)
    if spec.startswith("scripted:"):
        return ScriptedClient(load_script(Path(spec.removeprefix("scripted:"))))
    raise ValueError(f"unknown LLM spec: {spec!r} (expected ollama:<model>, openai:<model> or scripted:<file>)")
