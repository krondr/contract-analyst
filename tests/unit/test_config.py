from contract_analyst.config import load_settings


def test_defaults_and_overrides():
    s = load_settings()
    assert s.llm_model == "ollama:qwen3:8b" and s.max_reflection_rounds == 2
    assert "core_terms" in s.field_groups and "dealer" in s.field_groups
    assert load_settings(llm_model="ollama:qwen3:14b").llm_model == "ollama:qwen3:14b"


def test_every_llm_call_is_bounded_by_default():
    s = load_settings()
    # A capped answer plus a full clause batch still fits the context window, so the cap ends a runaway answer
    # before Ollama starts shifting the context.
    assert s.llm_max_tokens + s.llm_batch_tokens + 2000 <= s.llm_num_ctx
    assert 0 < s.llm_think_max_tokens < s.llm_num_ctx and s.llm_keep_alive and s.llm_call_deadline > 0


def test_every_grouped_field_has_a_description():
    s = load_settings()
    fields = {f for group in s.field_groups.values() for f in group}
    assert fields <= set(s.field_descriptions)


def test_langsmith_tracing_is_forced_off_before_langchain_loads():
    import os
    import subprocess
    import sys

    probe = ("import sys, os\n"
             "import contract_analyst.cli\n"
             "assert 'langchain_core' not in sys.modules\n"
             "from langchain_core.tracers.context import _tracing_v2_is_enabled\n"
             "import contract_analyst.agent.runner\n"
             "print(_tracing_v2_is_enabled(), os.environ.get('LANGSMITH_API_KEY'), os.environ.get('LANGCHAIN_API_KEY'))")
    env = os.environ | {"LANGSMITH_TRACING": "true", "LANGCHAIN_TRACING_V2": "true", "LANGSMITH_API_KEY": "lsv2-x",
                        "LANGCHAIN_API_KEY": "lsv2-x"}
    out = subprocess.run([sys.executable, "-c", probe], env=env, capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["False", "None", "None"]


def test_env_overrides_ollama_url(monkeypatch):
    monkeypatch.setenv("CA_OLLAMA_URL", "http://127.0.0.1:9")
    assert load_settings().ollama_url == "http://127.0.0.1:9"
