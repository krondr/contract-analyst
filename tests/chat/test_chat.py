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


def test_memory_keeps_only_what_the_user_saw(tmp_path):
    client = ScriptedClient([call("get_contract_terms", contract_number=MAIN),
                             AssistantTurn(content="Ставка составляет 900 рублей."),
                             AssistantTurn(content="Да, договор расторгнут.")])
    session = ChatSession(kb(tmp_path), client, load_settings(), tmp_path / "chat.sqlite")
    assert session.ask("Какая ставка?", "t10") == "Нет данных в базе знаний."
    session.ask("А он расторгнут?", "t10")
    context = client.calls[2]["messages"][1:]
    assert [(m["role"], m["content"]) for m in context] == [
        ("user", "Какая ставка?"), ("assistant", "Нет данных в базе знаний."), ("user", "А он расторгнут?")]


def test_tools_cite_status_documents_and_contracts(tmp_path):
    base = kb(tmp_path)
    terms = json.loads(run_chat_tool(base, "get_contract_terms", {"contract_number": MAIN}))
    assert terms["status"] == "действует [doc:m:p1]"
    assert len(terms["documents"]) == 2 and all(d.endswith(("[doc:m:p1]", "[doc:s:p1]")) for d in terms["documents"])
    (contract,) = json.loads(run_chat_tool(base, "list_contracts", {}))["contracts"]
    assert contract["source"] == "[doc:m:p1]"


def test_status_of_a_contract_not_yet_in_force_reads_as_a_future_start(tmp_path):
    args = {"contract_number": MAIN, "as_of": "2020-06-01"}
    terms = json.loads(run_chat_tool(kb(tmp_path), "get_contract_terms", args))
    assert terms["status"] == "ещё не вступил в силу; вступает в силу 2021-01-21 [doc:m:p1]"


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


def _chat_eval(tmp_path, questions, answers):
    path = tmp_path / "q.yaml"
    path.write_text(questions, encoding="utf-8")
    client = ScriptedClient([AssistantTurn(content=a) for a in answers])
    return chat_eval(kb(tmp_path).root, path, client, load_settings(), tmp_path / "e.sqlite")


def test_only_the_refusal_itself_counts_as_a_refusal(tmp_path):
    scores = _chat_eval(tmp_path, """
- {id: q1, question: "Какая ставка?", expected_facts: ["500"], unanswerable: false, follow_up_of: null}
- {id: q2, question: "Кто директор?", expected_facts: [], unanswerable: true, follow_up_of: null}
- {id: q3, question: "Какой штраф?", expected_facts: [], unanswerable: true, follow_up_of: null}
""", ["Ставка 500 рублей [doc:m:p1]. Нет данных о НДС.", "Нет данных о директоре, но ставка 500 рублей [doc:m:p1].",
      "«Нет данных в базе знаний»"])
    assert scores.accuracy == 2 / 3 and scores.refusal_precision == 1.0 and scores.with_refs_share == 1.0


def test_answerable_question_without_facts_is_not_scored(tmp_path):
    import pytest

    with pytest.warns(UserWarning, match="q1"):
        scores = _chat_eval(tmp_path, """
- {id: q1, question: "Что известно?", expected_facts: [], unanswerable: false, follow_up_of: null}
- {id: q2, question: "Какая ставка?", expected_facts: ["500"], unanswerable: false, follow_up_of: null}
""", ["Есть договор [doc:m:p1].", "Ставка 700 рублей [doc:m:p1]."])
    assert scores.accuracy == 0.0 and scores.invalid == ["q1"]


def test_llm_outage_during_a_question_is_an_answer_not_a_crash(tmp_path):
    from contract_analyst.chat.graph import ChatSession
    from contract_analyst.config import load_settings
    from contract_analyst.kb.store import KnowledgeBase

    class Down:
        model_name, usage = "down", {"calls": 0}

        def chat(self, messages, tools=None, think=False):
            raise ConnectionError("connection refused")

    session = ChatSession(KnowledgeBase(tmp_path / "kb"), Down(), load_settings(), tmp_path / "memory.sqlite")
    answer = session.ask("Какая ставка действует?", "t1")
    assert "ollama serve" in answer


def test_rates_reach_the_chat_with_vat_in_words(tmp_path):
    terms = json.loads(run_chat_tool(kb(tmp_path), "get_contract_terms", {"contract_number": MAIN}))
    assert terms["rates"] and all("excluded" not in r and "included" not in r and "unknown" not in r
                                  for r in terms["rates"] + terms.get("planned_rates", []))
    assert any("НДС" in r for r in terms["rates"])


def test_expected_dates_and_amounts_match_in_any_written_form():
    from contract_analyst.eval.chat_eval import fact_present

    assert fact_present("01.01.2027", "ставка 800 руб. с 2027-01-01 [doc:x:p1]")
    assert fact_present("01.04.2025", "с «1» апреля 2025 г. плата 120 000 руб.")
    assert fact_present("120 000", "плата 120000 руб.")
    assert not fact_present("01.01.2027", "ставка 800 руб. с 2026-01-01")


def test_rate_lines_never_print_none_or_vat_for_percent(tmp_path):
    from decimal import Decimal

    from contract_analyst.chat.tools import _rate_line
    from contract_analyst.models import RateRow, ValueStatus

    planned = RateRow(label="Ставка", amount=Decimal(800), status=ValueStatus.ACCEPTED, vat="excluded")
    percent = RateRow(label="Вознаграждение", amount=Decimal(60), unit="%", status=ValueStatus.ACCEPTED)
    assert "None" not in _rate_line(planned) and "без НДС" in _rate_line(planned)
    assert "НДС" not in _rate_line(percent) and "60 %" in _rate_line(percent)
