from typer.testing import CliRunner

from contract_analyst.cli import app


def test_cli_exits_2_with_hint_when_ollama_down(tmp_path, monkeypatch, fixtures_package):
    monkeypatch.setenv("CA_OLLAMA_URL", "http://127.0.0.1:9")
    res = CliRunner().invoke(app, ["analyze", str(fixtures_package), "--model", "ollama:qwen3:8b", "--out", str(tmp_path)])
    assert res.exit_code == 2 and "ollama serve" in res.output and "Traceback" not in res.output
