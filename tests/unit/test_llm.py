import json

import pytest
from pydantic import BaseModel

from contract_analyst.config import load_settings
from contract_analyst.llm.cache import CachingClient
from contract_analyst.llm.client import AssistantTurn, LLMUnavailable, ReplayMiss, ToolCall
from contract_analyst.llm.factory import get_llm_client
from contract_analyst.llm.ollama_client import OllamaClient
from contract_analyst.llm.prompts import load_prompt
from contract_analyst.llm.scripted import ScriptedClient, ScriptExhausted, load_script
from contract_analyst.llm.structured import StructuredFailure, call_structured


class A(BaseModel):
    a: int


M = [{"role": "user", "content": "q"}]


def test_structured_retries_then_succeeds():
    client = ScriptedClient(["not json", '{"a": 1}'])
    assert call_structured(client, M, A).a == 1
    assert "Ответ не прошёл проверку схемы" in client.calls[1]["messages"][-1]["content"]


def test_structured_accepts_fenced_json():
    assert call_structured(ScriptedClient(['```json\n{"a": 2}\n```']), M, A).a == 2


def test_structured_gives_up_after_two_retries():
    r = call_structured(ScriptedClient(["x", "y", "z"]), M, A)
    assert isinstance(r, StructuredFailure) and r.raw == ["x", "y", "z"] and len(r.errors) == 3


def test_scripted_client_runs_out():
    with pytest.raises(ScriptExhausted):
        ScriptedClient([]).chat(M)


def test_cache_record_then_replay(tmp_path):
    db = tmp_path / "c.sqlite"
    CachingClient(ScriptedClient(['{"a": 1}']), db, "record").structured(M, A)
    assert CachingClient(ScriptedClient([]), db, "replay").structured(M, A) == '{"a": 1}'


def test_cache_key_depends_on_messages_and_tools(tmp_path):
    db = tmp_path / "c.sqlite"
    turn = AssistantTurn(content="Мысль", tool_calls=[ToolCall(id="1", name="list_inputs", args={"thought": "t"})])
    rec = CachingClient(ScriptedClient([turn, AssistantTurn(content="другое")]), db, "record")
    assert rec.chat(M, tools=[{"name": "x"}]) == turn
    assert rec.chat(M, tools=[{"name": "y"}]).content == "другое"
    assert CachingClient(ScriptedClient([]), db, "replay").chat(M, tools=[{"name": "x"}]) == turn


def test_replay_miss_raises(tmp_path):
    with pytest.raises(ReplayMiss):
        CachingClient(ScriptedClient([]), tmp_path / "c.sqlite", "replay").chat(M)


def test_ollama_unavailable_gives_hint():
    with pytest.raises(LLMUnavailable, match="ollama serve"):
        OllamaClient("qwen3:8b", base_url="http://127.0.0.1:9", num_ctx=2048).check_available()


def test_load_prompt_picks_highest_version(tmp_path):
    (tmp_path / "x_v1.md").write_text("a")
    (tmp_path / "x_v2.md").write_text("b {{ n }}")
    (tmp_path / "x_v10.md").write_text("c {{ n }}")
    p = load_prompt("x", tmp_path)
    assert (p.version, p.render(n=3)) == (10, "c 3")


def test_load_script_formats(tmp_path):
    path = tmp_path / "s.json"
    path.write_text(json.dumps([{"structured": {"a": 1}}, "raw", {"chat": {"content": "ok", "tool_calls": []}}]))
    script = load_script(path)
    assert script[0] == '{"a": 1}' and script[1] == "raw" and script[2] == AssistantTurn(content="ok")


def test_factory_builds_cached_ollama_client(tmp_path):
    client = get_llm_client(load_settings(llm_model="ollama:qwen3:8b", llm_cache_mode="replay"), tmp_path / "c.sqlite")
    assert isinstance(client, CachingClient) and client.model_name == "qwen3:8b"
    client.check_available()  # replay mode never needs a live server


def test_thinking_turns_are_cached_apart_and_keep_reasoning(tmp_path):
    db = tmp_path / "c.sqlite"
    plain, thinking = AssistantTurn(content="a"), AssistantTurn(content="b", reasoning="думаю")
    rec = CachingClient(ScriptedClient([plain, thinking]), db, "record")
    rec.chat(M)
    rec.chat(M, think=True)
    replay = CachingClient(ScriptedClient([]), db, "replay")
    assert replay.chat(M) == plain and replay.chat(M, think=True).reasoning == "думаю"


def test_ollama_chat_with_thinking_returns_reasoning(monkeypatch):
    from langchain_core.messages import AIMessage

    from contract_analyst.llm.ollama_client import OllamaClient

    class FakeLLM:
        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            return AIMessage("", additional_kwargs={"reasoning_content": "думаю"},
                             tool_calls=[{"name": "list_inputs", "args": {}, "id": "1"}])

    client = OllamaClient("qwen3:8b")
    assert client._get(think=True).reasoning is True and client._get().reasoning is False
    monkeypatch.setattr(client, "_get", lambda think=False: FakeLLM())
    turn = client.chat(M, tools=[{"name": "list_inputs"}], think=True)
    assert turn.reasoning == "думаю" and turn.tool_calls[0].name == "list_inputs"
