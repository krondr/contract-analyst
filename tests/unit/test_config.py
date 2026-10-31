from contract_analyst.config import load_settings


def test_defaults_and_overrides():
    s = load_settings()
    assert s.llm_model == "ollama:qwen3:8b" and s.max_reflection_rounds == 2
    assert "core_terms" in s.field_groups and "dealer" in s.field_groups
    assert load_settings(llm_model="ollama:qwen3:14b").llm_model == "ollama:qwen3:14b"


def test_every_grouped_field_has_a_description():
    s = load_settings()
    fields = {f for group in s.field_groups.values() for f in group}
    assert fields <= set(s.field_descriptions)


def test_env_overrides_ollama_url(monkeypatch):
    monkeypatch.setenv("CA_OLLAMA_URL", "http://127.0.0.1:9")
    assert load_settings().ollama_url == "http://127.0.0.1:9"
