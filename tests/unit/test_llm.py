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


@pytest.mark.parametrize("url", ["http://localhost:11434", "http://127.0.0.1:11434", "http://[::1]:11434",
                                 "127.0.0.2:11434"])
def test_loopback_llm_hosts_need_no_permission(url, monkeypatch):
    monkeypatch.delenv("CA_ALLOW_REMOTE_LLM", raising=False)
    OllamaClient("qwen3:8b", base_url=url)


@pytest.mark.parametrize("url", ["http://gpu-box:11434", "http://10.0.0.5:11434", "http://0.0.0.0:11434",
                                 "https://llm.example.com"])
def test_remote_llm_host_needs_explicit_permission(url, monkeypatch):
    monkeypatch.delenv("CA_ALLOW_REMOTE_LLM", raising=False)
    with pytest.raises(LLMUnavailable, match="CA_ALLOW_REMOTE_LLM=1"):
        OllamaClient("qwen3:8b", base_url=url)
    monkeypatch.setenv("CA_ALLOW_REMOTE_LLM", "1")
    OllamaClient("qwen3:8b", base_url=url)


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


def test_factory_passes_the_caps_from_settings(tmp_path):
    settings = load_settings(llm_model="ollama:qwen3:8b", llm_max_tokens=1000, llm_think_max_tokens=2000,
                             llm_keep_alive="5m", llm_call_deadline=60)
    inner = get_llm_client(settings, tmp_path / "c.sqlite").inner
    assert (inner.max_tokens, inner.think_max_tokens, inner.keep_alive, inner.deadline) == (1000, 2000, "5m", 60)


def test_thinking_turns_are_cached_apart_and_keep_reasoning(tmp_path):
    db = tmp_path / "c.sqlite"
    plain, thinking = AssistantTurn(content="a"), AssistantTurn(content="b", reasoning="думаю")
    rec = CachingClient(ScriptedClient([plain, thinking]), db, "record")
    rec.chat(M)
    rec.chat(M, think=True)
    replay = CachingClient(ScriptedClient([]), db, "replay")
    assert replay.chat(M) == plain and replay.chat(M, think=True).reasoning == "думаю"


class FakeStreamingLLM:
    """Stands in for ChatOllama: streams the given chunks, records the request, notices when the stream is closed."""

    def __init__(self, chunks=(), endless: str | None = None, pause: float = 0.0):
        self.chunks, self.endless, self.pause = list(chunks), endless, pause
        self.requests: list[dict] = []
        self.closed = False

    def bind_tools(self, tools):
        return self

    def bind(self, **kwargs):
        return self

    def stream(self, messages, **kwargs):
        import time

        from langchain_core.messages import AIMessageChunk

        self.requests.append(kwargs)
        try:
            yield from self.chunks
            while self.endless is not None:
                time.sleep(self.pause)
                yield AIMessageChunk(content=self.endless)
        finally:
            self.closed = True


def _chunks(*texts, done_reason="stop", **extra):
    from langchain_core.messages import AIMessageChunk

    *head, last = texts
    return [AIMessageChunk(content=t) for t in head] + [AIMessageChunk(
        content=last, response_metadata={"done_reason": done_reason},
        usage_metadata={"input_tokens": 5, "output_tokens": 7, "total_tokens": 12}, **extra)]


def test_ollama_chat_with_thinking_returns_reasoning(monkeypatch):
    from langchain_core.messages import AIMessageChunk

    fake = FakeStreamingLLM([AIMessageChunk(content="", additional_kwargs={"reasoning_content": "ду"}),
                             AIMessageChunk(content="", additional_kwargs={"reasoning_content": "маю"},
                                            tool_calls=[{"name": "list_inputs", "args": {}, "id": "1"}],
                                            response_metadata={"done_reason": "stop"})])
    client = OllamaClient("qwen3:8b")
    assert client._get(think=True).reasoning is True and client._get().reasoning is False
    monkeypatch.setattr(client, "_get", lambda think=False: fake)
    turn = client.chat(M, tools=[{"name": "list_inputs"}], think=True)
    assert turn.reasoning == "думаю" and turn.tool_calls[0].name == "list_inputs"


def test_every_ollama_call_is_capped_and_thinking_uses_qwen3_sampling(monkeypatch):
    client = OllamaClient("qwen3:8b", max_tokens=4096, think_max_tokens=3000, keep_alive="30m")
    fake = FakeStreamingLLM(_chunks('{"a": 1}'))
    monkeypatch.setattr(client, "_get", lambda think=False: fake)
    client.structured(M, A)
    client.chat(M)
    client.chat(M, think=True)
    structured, plain, thinking = (r["options"] for r in fake.requests)
    assert structured["num_predict"] == plain["num_predict"] == 4096 and thinking["num_predict"] == 3000
    assert structured["temperature"] == plain["temperature"] == 0
    assert {k: thinking[k] for k in ("temperature", "top_p", "top_k", "min_p")} == {
        "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0}


def test_ollama_request_keeps_the_model_loaded_and_carries_the_cap():
    from langchain_core.messages import HumanMessage

    client = OllamaClient("qwen3:8b", max_tokens=4096, keep_alive="30m")
    params = client._get()._chat_params([HumanMessage("q")], options=client._options(think=False))
    assert params["keep_alive"] == "30m" and params["options"]["num_predict"] == 4096


def test_deadline_closes_the_stream_and_counts_as_timeout(monkeypatch):
    client = OllamaClient("qwen3:8b", deadline=0.2)
    fake = FakeStreamingLLM(endless="повтор ", pause=0.05)
    monkeypatch.setattr(client, "_get", lambda think=False: fake)
    with pytest.raises(TimeoutError):
        client.structured(M, A)
    assert fake.closed  # closing the stream closes the connection: that is what makes Ollama stop generating


def test_deadline_drops_the_connection_of_a_real_endless_stream():
    """Through the real ChatOllama/httpx stack: a loopback server streams Ollama NDJSON forever."""
    import threading
    import time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    disconnected = threading.Event()

    class Endless(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.end_headers()
            line = json.dumps({"model": "m", "message": {"role": "assistant", "content": "повтор "}, "done": False})
            try:
                while True:
                    self.wfile.write((line + "\n").encode())
                    self.wfile.flush()
                    time.sleep(0.02)
            except OSError:
                disconnected.set()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Endless)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        client = OllamaClient("m", base_url=f"http://127.0.0.1:{server.server_address[1]}", deadline=0.3)
        with pytest.raises(TimeoutError):
            client.structured(M, A)
        assert disconnected.wait(2)  # Ollama aborts a generation when its client goes away
    finally:
        server.shutdown()


def test_answer_cut_at_the_token_cap_is_counted(monkeypatch):
    client = OllamaClient("qwen3:8b")
    monkeypatch.setattr(client, "_get", lambda think=False: FakeStreamingLLM(_chunks('{"a": [1, ', '1, 1', done_reason="length")))
    assert client.structured(M, A) == '{"a": [1, 1, 1'
    assert client.usage["truncated"] == 1 and client.usage["output_tokens"] == 7


def test_truncated_answer_is_retried_once_shorter_without_the_cut_text():
    client = ScriptedClient(['{"a": [1, 1, 1, 1', '{"a": 1}'])
    assert call_structured(client, M, A).a == 1
    retry = client.calls[1]["messages"]
    assert retry[:-1] == M and "короче" in retry[-1]["content"] and "[1, 1" not in json.dumps(retry, ensure_ascii=False)


def test_answer_truncated_twice_is_a_truncation_failure_not_a_third_try():
    client = ScriptedClient(['{"a": [1, 1', '{"a": [1, 1, 1', '{"a": 1}'])
    result = call_structured(client, M, A)
    assert isinstance(result, StructuredFailure) and result.truncated and len(client.calls) == 2
