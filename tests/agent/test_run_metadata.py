"""metrics.json names what produced the run: the commit of the code and the LLM generation parameters."""

import json
import re
from datetime import date

import pytest

from contract_analyst.agent import steps
from contract_analyst.agent.runner import run_analysis
from contract_analyst.config import load_settings
from contract_analyst.llm.cache import CachingClient
from contract_analyst.llm.scripted import ScriptedClient
from tests.agent.test_hybrid import RECORDS
from tests.fixtures.builders import FakeOcrEngine
from tests.fixtures.builders import make_record as rec

SUMMARY = '{"summary": "Аренда."}'


@pytest.fixture(autouse=True)
def stub_understanding(monkeypatch):
    def fake(doc, header, client, settings, groups=None):
        spec = dict(RECORDS[doc.file.name])
        return rec(doc.doc_id, spec.pop("number"), file_name=doc.file.name, **spec)

    monkeypatch.setattr(steps, "understand_document", fake)


class ParamsClient(ScriptedClient):
    """A scripted client with generation parameters, as the real clients have."""

    def cache_params(self) -> dict:
        return {"max_tokens": 1000, "sampling": {"temperature": 0.0}}


def metrics(tmp_path, package, client) -> dict:
    r = run_analysis(package, load_settings(), date(2026, 10, 30), "A", client=client, ocr_engine=FakeOcrEngine(),
                     out_dir=tmp_path / "runs", read_cache=None, console=False)
    return json.loads((r.run_dir / "metrics.json").read_text(encoding="utf-8"))


def test_metrics_name_the_commit_of_the_code_and_the_generation_params(tmp_path, fixtures_package, monkeypatch):
    monkeypatch.chdir(tmp_path)  # the commit of the code that ran, not of the folder the run was started in
    m = metrics(tmp_path, fixtures_package, CachingClient(ParamsClient([SUMMARY]), tmp_path / "c.sqlite", "record"))
    assert re.fullmatch(r"[0-9a-f]{40}", m["git_commit"] or "")
    assert m["llm_params"] == {"max_tokens": 1000, "sampling": {"temperature": 0.0}}


@pytest.mark.parametrize("env", ["PATH", "GIT_DIR"], ids=["no git", "no repository"])
def test_without_git_the_commit_is_null_and_the_run_still_ends(env, tmp_path, fixtures_package, monkeypatch):
    monkeypatch.setenv(env, str(tmp_path / "missing"))
    m = metrics(tmp_path, fixtures_package, ScriptedClient([SUMMARY]))
    assert m["git_commit"] is None
    assert m["llm_params"] == {}  # a client without generation parameters
