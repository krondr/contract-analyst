"""Input selection: --doc and --chain must match what the user means, or the run fails with a message."""

import unicodedata
from datetime import date
from pathlib import Path

import pytest

from contract_analyst.agent import steps
from contract_analyst.agent.runner import InputError, run_analysis
from contract_analyst.config import load_settings
from contract_analyst.llm.client import AssistantTurn
from contract_analyst.llm.scripted import ScriptedClient
from tests.fixtures.builders import FakeOcrEngine, make_text_pdf
from tests.fixtures.builders import make_record as rec

NAME = "Договор й.pdf"  # "й" differs between NFC and NFD
SCRIPT = ['{"items": ["План"]}', AssistantTurn(content="Готово"), '{"goal_met": true, "gaps": []}',
          AssistantTurn(content="Готово"), '{"goal_met": true, "gaps": []}', '{"summary": "x"}']


@pytest.fixture
def package(tmp_path, monkeypatch):
    pkg = tmp_path / "pkg + архив"
    pkg.mkdir()
    make_text_pdf(pkg / unicodedata.normalize("NFD", NAME), ["Договор № D990000001-01\nг. Тверь 10.01.2024 г."])
    make_text_pdf(pkg / "Другой.pdf", ["Договор № D990000009-01\nг. Псков 01.01.2023 г."])
    monkeypatch.setattr(steps, "understand_document", lambda doc, header, client, settings, groups=None: rec(
        doc.doc_id, header.number, kind="main", file_name=doc.file.name))
    return pkg


def run(package, tmp_path, **kw):
    return run_analysis(package, load_settings(), date(2026, 10, 30), client=ScriptedClient(list(SCRIPT)),
                        ocr_engine=FakeOcrEngine(), out_dir=tmp_path / "runs", read_cache=None, console=False, **kw)


def analyzed(result) -> list[str]:
    report = result.report_path.read_text(encoding="utf-8")
    return [n for n in ("D990000001-01", "D990000009-01") if f"### {n}" in report]


def test_doc_typed_in_nfc_matches_a_file_named_in_nfd(package, tmp_path):
    assert analyzed(run(package, tmp_path, only_doc=package / unicodedata.normalize("NFC", NAME))) == ["D990000001-01"]


def test_absolute_doc_matches_a_relative_input_folder(package, tmp_path, monkeypatch):
    monkeypatch.chdir(package.parent)
    r = run(Path(package.name), tmp_path, only_doc=(package / "Другой.pdf").resolve())
    assert analyzed(r) == ["D990000009-01"]


def test_doc_may_be_given_relative_to_the_input_folder(package, tmp_path):
    assert analyzed(run(package, tmp_path, only_doc=Path("Другой.pdf"))) == ["D990000009-01"]


@pytest.mark.parametrize("case", ["missing_folder", "empty_folder", "doc", "chain"])
def test_selection_that_matches_nothing_fails_with_a_message(package, tmp_path, case):
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / ".DS_Store").write_bytes(b"x")
    args = {"missing_folder": (tmp_path / "нет", {}), "empty_folder": (empty, {}),
            "doc": (package, {"only_doc": package / "ДС9.pdf"}), "chain": (package, {"only_chain": "D990000077-01"})}
    folder, kw = args[case]
    with pytest.raises(InputError) as exc:
        run(folder, tmp_path, **kw)
    assert str(exc.value)


def test_cli_reports_a_missing_folder_and_exits_2(tmp_path):
    from typer.testing import CliRunner

    from contract_analyst.cli import app

    res = CliRunner().invoke(app, ["analyze", str(tmp_path / "нет"), "--out", str(tmp_path / "runs")])
    assert res.exit_code == 2 and "не найдена" in res.output
