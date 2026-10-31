from datetime import date

import pytest

from contract_analyst.agent.context import RunContext
from contract_analyst.agent.trace import TraceLogger
from contract_analyst.config import load_settings
from contract_analyst.llm.scripted import ScriptedClient
from tests.fixtures.builders import FakeOcrEngine


def make_ctx(tmp_path, package, client=None, ocr=None, **settings):
    return RunContext(settings=load_settings(**settings), client=client or ScriptedClient([]),
                      ocr_engine=ocr or FakeOcrEngine(), trace=TraceLogger(tmp_path / "run", "test", console=False),
                      input_dir=package, run_dir=tmp_path / "run", as_of=date(2026, 10, 30))


@pytest.fixture
def ctx(tmp_path, fixtures_package):
    return make_ctx(tmp_path, fixtures_package)
