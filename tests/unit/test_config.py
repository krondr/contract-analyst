from contract_analyst.config import load_settings


def test_defaults_and_overrides():
    s = load_settings()
    assert s.llm_model == "ollama:qwen3:8b" and s.max_reflection_rounds == 2
    assert "core_terms" in s.field_groups and "dealer" in s.field_groups
    assert load_settings(llm_model="ollama:qwen3:14b").llm_model == "ollama:qwen3:14b"


def test_code_defaults_equal_the_yaml_for_every_key(monkeypatch):
    import os

    import yaml

    from contract_analyst.config import DEFAULT_CONFIG, Settings

    for name in [n for n in os.environ if n.startswith("CA_")]:
        monkeypatch.delenv(name)
    keys = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    assert set(keys) <= set(Settings.model_fields)
    defaults, from_yaml = Settings(), load_settings()
    assert {k: getattr(defaults, k) for k in keys} == {k: getattr(from_yaml, k) for k in keys}


def test_installed_without_the_repo_settings_and_prompts_come_from_the_package(tmp_path):
    """Lay the package out as the wheel does (the module plus what pyproject.toml force-includes), with no config/
    or prompts/ at the would-be repo root, and load both in a clean interpreter."""
    import json
    import shutil
    import subprocess
    import sys
    import tomllib

    from contract_analyst.config import REPO_ROOT
    from contract_analyst.llm.prompts import load_prompt

    site = tmp_path / "site-packages"
    shutil.copytree(REPO_ROOT / "src" / "contract_analyst", site / "contract_analyst",
                    ignore=shutil.ignore_patterns("__pycache__"))
    build = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["hatch"]["build"]
    for source, target in build["targets"]["wheel"]["force-include"].items():
        shutil.copytree(REPO_ROOT / source, site / target)
    probe = ("import json, sys; sys.path.insert(0, sys.argv[1])\n"
             "from contract_analyst.config import REPO_ROOT, load_settings\n"
             "from contract_analyst.llm.prompts import load_prompt\n"
             "print(json.dumps({'root': str(REPO_ROOT), 'settings': load_settings().model_dump(mode='json'),\n"
             "                  'chat': load_prompt('chat').model_dump()}))")
    out = subprocess.run([sys.executable, "-I", "-c", probe, str(site)], cwd=tmp_path, capture_output=True,
                         text=True, check=False)
    assert out.returncode == 0, out.stderr
    loaded = json.loads(out.stdout)
    assert loaded["root"] == str(tmp_path.resolve()) and not (tmp_path / "config").exists()
    assert loaded["settings"] == load_settings().model_dump(mode="json")
    assert loaded["chat"] == load_prompt("chat").model_dump()


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


def test_server_config_file_overrides_the_defaults(tmp_path, monkeypatch):
    from contract_analyst.config import load_settings

    extra = tmp_path / "server.yaml"
    extra.write_text("openai_base_url: https://llm.example/v1\nopenai_extra_body: {reasoning: {enabled: false}}\n",
                     encoding="utf-8")
    monkeypatch.setenv("CA_CONFIG", str(extra))
    s = load_settings()
    assert s.openai_base_url == "https://llm.example/v1" and s.openai_extra_body == {"reasoning": {"enabled": False}}
    assert s.llm_max_tokens == 4096
