import json

from contract_analyst.chat.graph import ChatSession
from contract_analyst.chat.tools import run_chat_tool
from contract_analyst.config import load_settings
from contract_analyst.eval.chat_eval import chat_eval
from contract_analyst.kb.store import KnowledgeBase
from contract_analyst.llm.client import AssistantTurn, ToolCall
from contract_analyst.llm.scripted import ScriptedClient
from tests.unit.test_kb import MAIN, small_run


def kb(tmp_path):
    base = KnowledgeBase(tmp_path / "kb")
    base.save(**small_run())
    return base


def call(name, **args):
    return AssistantTurn(tool_calls=[ToolCall(id=name, name=name, args={"thought": "ищу", **args})])


def test_fact_answer_cites_source(tmp_path):
    client = ScriptedClient([call("get_contract_terms", contract_number=MAIN),
                             AssistantTurn(content="Действует ставка 500 рублей [doc:m:p1].")])
    session = ChatSession(kb(tmp_path), client, load_settings(), tmp_path / "chat.sqlite")
    answer = session.ask("Какая ставка действует по договору?", thread="t1")
    assert answer == "Действует ставка 500 рублей [doc:m:p1]."
    tool_message = client.calls[1]["messages"][-1]
    assert tool_message["role"] == "tool" and "[doc:m:p1]" in tool_message["content"]


def test_follow_up_sees_previous_turn(tmp_path):
    client = ScriptedClient([AssistantTurn(content="Договор один: D990000001-01 [doc:m:p1]."),
                             AssistantTurn(content="Он действует [doc:m:p1].")])
    session = ChatSession(kb(tmp_path), client, load_settings(), tmp_path / "chat.sqlite")
    session.ask("Сколько договоров?", thread="t2")
    session.ask("А он действует?", thread="t2")
    second = [m["content"] for m in client.calls[1]["messages"] if m["role"] == "user"]
    assert second == ["Сколько договоров?", "А он действует?"]


def test_memory_survives_a_new_session_object(tmp_path):
    base = kb(tmp_path)
    ChatSession(base, ScriptedClient([AssistantTurn(content="Привет.")]), load_settings(), tmp_path / "c.sqlite").ask("Привет", "t3")
    client = ScriptedClient([AssistantTurn(content="Помню.")])
    ChatSession(base, client, load_settings(), tmp_path / "c.sqlite").ask("Что я спросил?", "t3")
    assert "Привет" in json.dumps(client.calls[0]["messages"], ensure_ascii=False)


def test_unsupported_number_removed_and_empty_answer_becomes_refusal(tmp_path):
    session = ChatSession(kb(tmp_path), ScriptedClient([AssistantTurn(content="Срок истекает в 2030 году.")]),
                          load_settings(), tmp_path / "chat.sqlite")
    assert session.ask("Когда истекает срок?", thread="t4") == "Нет данных в базе знаний."


def test_number_absent_from_the_cited_page_removed(tmp_path):
    session = ChatSession(kb(tmp_path), ScriptedClient([AssistantTurn(content="Действует ставка 900 рублей [doc:m:p1].")]),
                          load_settings(), tmp_path / "chat.sqlite")
    assert session.ask("Какая ставка действует?", thread="t5") == "Нет данных в базе знаний."


def empty_kb(tmp_path):
    base = KnowledgeBase(tmp_path / "empty")
    base.save(documents={}, records={}, chains=[], consolidated={}, clusters=[], decisions=[])
    return base


def test_statement_without_reference_refused_even_without_numbers(tmp_path):
    session = ChatSession(empty_kb(tmp_path), ScriptedClient([AssistantTurn(content="Да, расторгнут.")]),
                          load_settings(), tmp_path / "chat.sqlite")
    assert session.ask("Договор расторгнут?", thread="t6") == "Нет данных в базе знаний."


def test_only_referenced_statements_kept(tmp_path):
    client = ScriptedClient([AssistantTurn(content="Ставка 500 рублей [doc:m:p1]. Договор скоро расторгнут.")])
    session = ChatSession(kb(tmp_path), client, load_settings(), tmp_path / "chat.sqlite")
    assert session.ask("Какая ставка?", thread="t7") == "Ставка 500 рублей [doc:m:p1]."


def test_date_the_user_asked_about_frames_the_answer(tmp_path):
    client = ScriptedClient([AssistantTurn(content="На 30.10.2026 действует ставка 500 рублей [doc:m:p1]."),
                             AssistantTurn(content="Ставка 500 рублей действует с 30.10.2026 [doc:m:p1].")])
    session = ChatSession(kb(tmp_path), client, load_settings(), tmp_path / "chat.sqlite")
    question = "Какая ставка действует на 30.10.2026?"
    assert session.ask(question, thread="t8") == "На 30.10.2026 действует ставка 500 рублей [doc:m:p1]."
    assert session.ask(question, thread="t9") == "Нет данных в базе знаний."  # "с 30.10.2026" is a claim


def test_tools_cite_status_documents_and_contracts(tmp_path):
    base = kb(tmp_path)
    terms = json.loads(run_chat_tool(base, "get_contract_terms", {"contract_number": MAIN}))
    assert terms["status"] == "действует [doc:m:p1]"
    assert len(terms["documents"]) == 2 and all(d.endswith(("[doc:m:p1]", "[doc:s:p1]")) for d in terms["documents"])
    (contract,) = json.loads(run_chat_tool(base, "list_contracts", {}))["contracts"]
    assert contract["source"] == "[doc:m:p1]"


def test_chat_eval_scores_facts_refusals_and_refs(tmp_path):
    questions = tmp_path / "q.yaml"
    questions.write_text("""
- {id: q1, question: "Какая ставка?", expected_facts: ["500"], unanswerable: false, follow_up_of: null}
- {id: q2, question: "Кто директор?", expected_facts: [], unanswerable: true, follow_up_of: null}
""", encoding="utf-8")
    client = ScriptedClient([AssistantTurn(content="Ставка 500 рублей [doc:m:p1]."),
                             AssistantTurn(content="Нет данных в базе знаний.")])
    scores = chat_eval(kb(tmp_path).root, questions, client, load_settings(), tmp_path / "e.sqlite")
    assert scores.accuracy == 1.0 and scores.refusal_precision == 1.0 and scores.with_refs_share == 1.0
