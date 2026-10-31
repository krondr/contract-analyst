"""Fault injection: unreadable inputs and LLM failures must end in a report, never in a crash."""

import shutil
from datetime import date

import cv2
import numpy as np
import pymupdf
import pytest

from contract_analyst.agent import steps
from contract_analyst.agent.runner import run_analysis
from contract_analyst.agent.tools import TOOLS, ToolSpec
from contract_analyst.agent.trace import read_trace
from contract_analyst.config import load_settings
from contract_analyst.llm.client import AssistantTurn, ToolCall, new_usage
from tests.fixtures.builders import FakeOcrEngine


class FaultyClient:
    """Chat follows a short script then says it is done; structured answers are broken in a chosen way."""

    model_name = "faulty"

    def __init__(self, structured_fault: str = "invalid_json", chat_script=None):
        self.fault, self.usage = structured_fault, new_usage()
        self.turns = list(chat_script or [])

    def chat(self, messages, tools=None):
        self.usage["calls"] += 1
        return self.turns.pop(0) if self.turns else AssistantTurn(content="Готово")

    def structured(self, messages, schema):
        self.usage["calls"] += 1
        if self.fault == "timeout":
            raise TimeoutError("model did not answer in time")
        return "это не JSON"


def analyze_all(*names):
    return AssistantTurn(tool_calls=[ToolCall(id=str(i), name="analyze_document",
                                              args={"thought": "разбираю", "doc_id": n}) for i, n in enumerate(names)])


def run(package, tmp_path, client=None, ocr=None):
    return run_analysis(package, load_settings(), date(2026, 10, 30), client=client or FaultyClient(),
                        ocr_engine=ocr or FakeOcrEngine(), out_dir=tmp_path / "runs", read_cache=None, console=False)


def section(report_path, title, next_title):
    return report_path.read_text(encoding="utf-8").split(title)[1].split(next_title)[0]


@pytest.fixture
def package(tmp_path, fixtures_package):
    target = tmp_path / "pkg"
    shutil.copytree(fixtures_package, target)
    return target


def _errors(result):
    return [e for e in read_trace(result.trace_path) if e["event"] == "error"]


def test_empty_pdf(package, tmp_path):
    blank = pymupdf.open()
    blank.new_page()
    blank.save(package / "пустой.pdf")
    r = run(package, tmp_path)
    assert "пустой.pdf" in section(r.report_path, "## 6. Не обработано", "## 7.")
    assert any(e.get("file") == "пустой.pdf" for e in _errors(r))


@pytest.mark.parametrize("name,reason", [("битый.pdf", "corrupted"), ("закрытый.pdf", "encrypted"),
                                         ("таблица.xlsx", "unsupported_format")])
def test_unreadable_files(package, tmp_path, name, reason):
    if reason == "corrupted":
        (package / name).write_bytes(b"%PDF-broken")
    elif reason == "encrypted":
        doc = pymupdf.open()
        doc.new_page()
        doc.save(package / name, encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="u", owner_pw="o")
    else:
        (package / name).write_bytes(b"x")
    r = run(package, tmp_path)
    unprocessed = section(r.report_path, "## 6. Не обработано", "## 7.")
    assert name in unprocessed and reason in unprocessed
    assert any(e.get("file") == name for e in _errors(r))


def test_noise_image_gives_empty_ocr(package, tmp_path):
    rng = np.random.default_rng(0)
    cv2.imwrite(str(package / "шум.png"), rng.integers(0, 255, (300, 300, 3), dtype=np.uint8))

    class EmptyOnNoise(FakeOcrEngine):
        def recognize(self, image):
            return super().recognize(image) if image.shape[0] > 400 else FakeOcrEngine(text="").recognize(image)

    r = run(package, tmp_path, ocr=EmptyOnNoise())
    assert "шум.png" in section(r.report_path, "## 6. Не обработано", "## 7.")


def test_invalid_json_from_llm(package, tmp_path):
    r = run(package, tmp_path, client=FaultyClient("invalid_json", [analyze_all("ДС1.pdf")]))
    assert "llm_invalid_json" in section(r.report_path, "## 5. Отказы и противоречия", "## 6.")
    assert r.report_path.exists()


def test_llm_timeout(package, tmp_path):
    r = run(package, tmp_path, client=FaultyClient("timeout", [analyze_all("ДС1.pdf")]))
    assert "llm_timeout" in section(r.report_path, "## 5. Отказы и противоречия", "## 6.")
    assert any("timeout" in str(e.get("message", "")).lower() for e in _errors(r))


def test_tool_exception_does_not_stop_the_run(package, tmp_path, monkeypatch):
    def boom(ctx, args):
        raise RuntimeError("сбой инструмента")

    monkeypatch.setitem(TOOLS, "build_chains", ToolSpec("build_chains", "x", TOOLS["build_chains"].args_model, boom))
    script = [AssistantTurn(tool_calls=[ToolCall(id="1", name="build_chains", args={"thought": "собираю"})])]
    r = run(package, tmp_path, client=FaultyClient("invalid_json", script))
    assert any("сбой инструмента" in str(e.get("message", "")) for e in _errors(r))
    assert "### D990000001-01" in r.report_path.read_text(encoding="utf-8")
    assert steps is not None
