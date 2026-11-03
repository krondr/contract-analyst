import json

import httpx
import pytest
from pydantic import BaseModel

from contract_analyst.llm.cache import CachingClient
from contract_analyst.llm.client import LLMUnavailable, ReplayMiss
from contract_analyst.llm.openai_client import GREEDY, THINKING

M = [{"role": "system", "content": "Ты — агент."}, {"role": "user", "content": "Опись: a1"}]


class Out(BaseModel):
    a: int


def client(handler, **kw):
    from contract_analyst.llm.openai_client import OpenAICompatClient

    return OpenAICompatClient("qwen3-8b", "http://localhost:8000/v1", transport=httpx.MockTransport(handler), **kw)


def answer(message, finish="stop", usage=None):
    return httpx.Response(200, json={"choices": [{"message": message, "finish_reason": finish}],
                                     "usage": usage or {"prompt_tokens": 10, "completion_tokens": 5}})


def test_chat_sends_tools_capped_and_greedy_and_maps_calls_and_reasoning():
    sent = {}

    def handler(request):
        sent["url"], sent["auth"] = str(request.url), request.headers.get("authorization")
        sent["body"] = json.loads(request.content)
        return answer({"content": "", "reasoning_content": "Сначала разберу опись.",
                       "tool_calls": [{"id": "c1", "type": "function",
                                       "function": {"name": "analyze_document", "arguments": '{"doc_id": "a1"}'}}]})

    tools = [{"type": "function", "function": {"name": "analyze_document", "parameters": {}}}]
    turn = client(handler, api_key="secret", max_tokens=1234).chat(M, tools)
    assert sent["url"] == "http://localhost:8000/v1/chat/completions" and sent["auth"] == "Bearer secret"
    body = sent["body"]
    assert body["model"] == "qwen3-8b" and body["tools"] == tools and body["max_tokens"] == 1234
    assert body["temperature"] == 0 and body["messages"] == M
    assert turn.tool_calls[0].name == "analyze_document" and turn.tool_calls[0].args == {"doc_id": "a1"}
    assert turn.reasoning == "Сначала разберу опись."


def test_thinking_turn_uses_its_own_cap_and_sampling():
    bodies = []
    c = client(lambda r: bodies.append(json.loads(r.content)) or answer({"content": "ok"}), think_max_tokens=777)
    c.chat(M, think=True)
    assert bodies[0]["max_tokens"] == 777 and bodies[0]["temperature"] == 0.6 and bodies[0]["top_p"] == 0.95


def test_history_with_tool_calls_is_sent_in_openai_form():
    bodies = []
    history = [*M, {"role": "assistant", "content": "", "reasoning": "думаю",
                    "tool_calls": [{"id": "c1", "name": "list_inputs", "args": {"thought": "смотрю"}}]},
               {"role": "tool", "tool_call_id": "c1", "name": "list_inputs", "content": '{"status": "ok"}'}]
    client(lambda r: bodies.append(json.loads(r.content)) or answer({"content": "Готово"})).chat(history)
    assistant, tool = bodies[0]["messages"][2:]
    assert assistant == {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "list_inputs", "arguments": '{"thought": "смотрю"}'}}]}
    assert tool == {"role": "tool", "tool_call_id": "c1", "content": '{"status": "ok"}'}


def test_structured_asks_for_the_schema_and_counts_usage_and_truncation():
    bodies = []
    c = client(lambda r: bodies.append(json.loads(r.content)) or answer({"content": '{"a": 1'}, finish="length",
                                                                         usage={"prompt_tokens": 7, "completion_tokens": 3}))
    assert c.structured(M, Out) == '{"a": 1'
    fmt = bodies[0]["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["schema"] == Out.model_json_schema() | {"additionalProperties": False}
    assert c.usage == {"calls": 1, "input_tokens": 7, "output_tokens": 3, "truncated": 1}


def test_server_error_is_a_transport_error_the_run_survives():
    from contract_analyst.llm.ollama_client import TRANSPORT_ERRORS

    with pytest.raises(TRANSPORT_ERRORS):
        client(lambda r: httpx.Response(503, text="overloaded")).chat(M)


def test_unavailable_server_wrong_key_or_missing_model_give_hints():
    def down(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(LLMUnavailable, match="не отвечает"):
        client(down).check_available()
    with pytest.raises(LLMUnavailable, match="CA_OPENAI_API_KEY"):
        client(lambda r: httpx.Response(401)).check_available()
    with pytest.raises(LLMUnavailable, match="qwen3-8b"):
        client(lambda r: httpx.Response(200, json={"data": [{"id": "other"}]})).check_available()
    client(lambda r: httpx.Response(200, json={"data": [{"id": "qwen3-8b"}]})).check_available()


def test_remote_server_needs_explicit_permission(monkeypatch):
    from contract_analyst.llm.openai_client import OpenAICompatClient

    monkeypatch.delenv("CA_ALLOW_REMOTE_LLM", raising=False)
    with pytest.raises(LLMUnavailable, match="CA_ALLOW_REMOTE_LLM"):
        OpenAICompatClient("m", "https://llm.example.com/v1")
    monkeypatch.setenv("CA_ALLOW_REMOTE_LLM", "1")
    OpenAICompatClient("m", "https://llm.example.com/v1")


def test_factory_builds_a_cached_openai_client(tmp_path, monkeypatch):
    from contract_analyst.config import load_settings
    from contract_analyst.llm.cache import CachingClient
    from contract_analyst.llm.factory import get_llm_client

    monkeypatch.setenv("CA_OPENAI_BASE_URL", "http://127.0.0.1:9000/v1")
    client_ = get_llm_client(load_settings(llm_model="openai:qwen3-8b"), tmp_path / "c.sqlite")
    assert isinstance(client_, CachingClient)
    assert client_.inner.base_url == "http://127.0.0.1:9000/v1" and client_.model_name == "openai:qwen3-8b"


def test_server_specific_fields_switch_thinking_per_call():
    bodies = []
    c = client(lambda r: bodies.append(json.loads(r.content)) or answer({"content": "{}"}),
               extra_body={"chat_template_kwargs": {"enable_thinking": False}},
               think_extra_body={"chat_template_kwargs": {"enable_thinking": True}})
    c.structured(M, Out)
    c.chat(M)
    c.chat(M, think=True)
    assert [b["chat_template_kwargs"]["enable_thinking"] for b in bodies] == [False, False, True]


def test_factory_passes_the_server_fields_from_settings(tmp_path):
    from contract_analyst.config import load_settings
    from contract_analyst.llm.factory import get_llm_client

    settings = load_settings(llm_model="openai:m", openai_extra_body={"think": False},
                             openai_think_extra_body={"think": True})
    inner = get_llm_client(settings, tmp_path / "c.sqlite").inner
    assert (inner.extra_body, inner.think_extra_body) == ({"think": False}, {"think": True})


def test_structured_schema_is_closed_at_every_level_for_openai():
    class Inner(BaseModel):
        value: str | None = None

    class Outer(BaseModel):
        items: list[Inner] = []
        note: Inner | None = None
        by_name: dict[str, Inner] = {}

    bodies = []
    client(lambda r: bodies.append(json.loads(r.content)) or answer({"content": "{}"})).structured(M, Outer)
    fmt = bodies[0]["response_format"]["json_schema"]
    assert fmt["strict"] is False

    def objects(node):
        if isinstance(node, dict):
            if "properties" in node:
                yield node
            for v in node.values():
                yield from objects(v)
        elif isinstance(node, list):
            for v in node:
                yield from objects(v)

    found = list(objects(fmt["schema"]))
    assert len(found) >= 2 and all(o.get("additionalProperties") is False for o in found)
    by_name = fmt["schema"]["properties"]["by_name"]
    assert isinstance(by_name["additionalProperties"], dict)  # a map keeps the schema of its values


@pytest.mark.parametrize("response", [httpx.Response(200, json={"error": {"message": "upstream provider failed"}}),
                                      httpx.Response(200, text="<html>bad gateway</html>")])
def test_ok_status_without_an_answer_is_a_transport_error(response):
    with pytest.raises(ConnectionError, match="choices"):
        client(lambda request: response).chat(M)


GEN = {"max_tokens": 1000, "think_max_tokens": 1000, "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
       "think_extra_body": {"chat_template_kwargs": {"enable_thinking": True}}}


def offline(request):
    pytest.fail("a replay never asks the server")


def _recorded_turn(db, think, **params):
    CachingClient(client(lambda r: answer({"content": "ok"}), **params), db, "record").chat(M, think=think)


@pytest.mark.parametrize("change, think", [({"max_tokens": 2000}, False), ({"think_max_tokens": 2000}, True),
                                           ({"extra_body": {}}, False), ({"think_extra_body": {}}, True)],
                         ids=["max_tokens", "think_max_tokens", "extra_body", "think_extra_body"])
def test_turns_recorded_with_other_generation_params_are_not_replayed(change, think, tmp_path):
    from contract_analyst.llm.openai_client import OpenAICompatClient

    db = tmp_path / "c.sqlite"
    _recorded_turn(db, think, **GEN)
    # The deadline, the address and the key never change what the model writes: the turn is replayed.
    same = OpenAICompatClient("qwen3-8b", "http://127.0.0.1:8000/v1", api_key="other", deadline=5,
                              transport=httpx.MockTransport(offline), **GEN)
    assert CachingClient(same, db, "replay").chat(M, think=think).content == "ok"
    with pytest.raises(ReplayMiss):
        CachingClient(client(offline, **(GEN | change)), db, "replay").chat(M, think=think)


@pytest.mark.parametrize("sampling, think", [(GREEDY, False), (THINKING, True)], ids=["greedy", "thinking"])
def test_turns_recorded_with_other_sampling_are_not_replayed(sampling, think, tmp_path, monkeypatch):
    db = tmp_path / "c.sqlite"
    _recorded_turn(db, think)
    monkeypatch.setitem(sampling, "temperature", 0.3)
    with pytest.raises(ReplayMiss):
        CachingClient(client(offline), db, "replay").chat(M, think=think)

