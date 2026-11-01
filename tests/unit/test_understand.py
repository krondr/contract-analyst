import json
from datetime import date
from decimal import Decimal
from pathlib import Path

from contract_analyst.config import load_settings
from contract_analyst.ingest.inventory import list_inputs
from contract_analyst.ingest.reader import read_document
from contract_analyst.llm.scripted import ScriptedClient, load_script
from contract_analyst.parsing.header import HeaderGuess, parse_header
from contract_analyst.understand.routing import estimate_tokens, select_clauses, select_term_clauses
from contract_analyst.understand.understand import understand_document
from tests.fixtures.builders import FakeOcrEngine, make_docx, text_doc

SCRIPTS = Path(__file__).parent / "scripts"
SUPPLEMENT = ("Дополнительное соглашение № D990000002-01\n"
              "к Договору № D990000001-01 от «21» января 2021 г.\n"
              "1. Изложить п. 3.1 Договора в следующей редакции: «3.1. Ставка вознаграждения "
              "составляет 700 рублей без НДС».\n2. Настоящее Соглашение вступает в силу с 01.07.2026.")


def _header_json(**overrides):
    base = {"kind": "main", "title": "", "number": {"value": None}, "doc_date": {"value": None},
            "city": {"value": None}, "parent_number": {"value": None}, "parent_date": {"value": None},
            "effective_from": {"value": None}, "parties": [], "subject_category": "other"}
    base.update(overrides)
    return json.dumps(base, ensure_ascii=False)


def _understand(doc, script, groups=None):
    return understand_document(doc, parse_header(doc.pages[0].text), ScriptedClient(script), load_settings(), groups)


def test_grounded_values_accepted_invented_rejected():
    rec = _understand(text_doc([SUPPLEMENT]), load_script(SCRIPTS / "understand_supplement.json"))
    assert rec.number.value == "D990000002-01" and rec.number.status == "accepted"
    assert rec.parent.value.number == "D990000001-01" and rec.parent.value.date_stated == date(2021, 1, 21)
    assert [r.amount for r in rec.rates if r.status == "accepted"] == [Decimal(700)]
    assert any(r.status == "rejected" for r in rec.rates)
    assert rec.ops[0].op == "replace_clause" and rec.ops[0].status == "accepted"
    assert rec.effective_from.value == date(2026, 7, 1) and rec.kind.value == "supplement"


def test_title_body_mismatch_issue():
    doc = text_doc([("Договор № D990000010-01\nг. Тверь 25.01.2021 г.\n"
                    "заключили настоящее Дополнительное соглашение к договору № D990000010-01 от «21» января 2021 г.")])
    rec = _understand(doc, [_header_json(kind="supplement")], groups=["header"])
    assert "title_body_mismatch" in [i.code for i in rec.issues]


def test_regex_and_llm_disagree_gives_conflict():
    doc = text_doc([("Дополнительное соглашение № D990000002-01\n"
                    "к Договору № D990000001-01 от «21» января 2021 г.\n"
                    "Договор № D990000001-01 заключён «25» января 2021 г.")])
    llm = _header_json(kind="supplement", parent_date={"value": "2021-01-25", "page": 1,
                                                       "quote": "Договор № D990000001-01 заключён «25» января 2021 г."})
    rec = _understand(doc, [llm], groups=["header"])
    assert rec.parent.status == "conflict" and rec.parent.alternatives


def test_long_document_batched_not_truncated():
    s = load_settings()
    doc = text_doc([f"{i}. Ставка {i} составляет {i * 10} рублей. " + "слово " * 400 for i in range(1, 121)])
    batches = select_clauses(doc, "money", s)
    assert sum(len(b) for b in batches) == 120 and len(batches) > 1
    assert all(estimate_tokens(b) <= s.llm_batch_tokens for b in batches)


def test_oversized_clause_is_split_not_dropped():
    s = load_settings(llm_batch_tokens=500)
    doc = text_doc(["1. Ставка вознаграждения " + "очень длинный текст " * 300])
    batches = select_clauses(doc, "money", s)
    assert len(batches) > 1 and all(estimate_tokens(b) <= 500 for b in batches)
    assert sum(len(c.text) for b in batches for c in b) >= len(doc.clauses[0].text) - len(batches)


def test_only_requested_group_called():
    understand_document(text_doc(["1. Ставка 700 рублей"]), HeaderGuess(title_kind="unknown"),
                        ScriptedClient(['{"rates": []}']), load_settings(), groups=["money"])


def test_unknown_category_becomes_other_and_draft_flag():
    doc = text_doc(["ПРОЕКТ\nДоговор № D990000003-01\nг. Тверь 19.08.2020 г."], name="Проект договора.pdf")
    rec = _understand(doc, [_header_json(subject_category="xyz")], groups=["header"])
    assert rec.subject_category.value == "other" and rec.is_draft


def test_invalid_json_keeps_regex_values_and_records_issue():
    rec = _understand(text_doc([SUPPLEMENT]), ["x", "y", "z"], groups=["header"])
    assert "llm_invalid_json" in [i.code for i in rec.issues]
    assert rec.number.value == "D990000002-01" and rec.number.status == "accepted"


def test_timeout_records_issue():
    class Slow(ScriptedClient):
        def structured(self, messages, schema):
            raise TimeoutError("timed out")

    rec = understand_document(text_doc([SUPPLEMENT]), parse_header(SUPPLEMENT), Slow([]), load_settings(), ["header"])
    assert "llm_timeout" in [i.code for i in rec.issues]


CHANGE_LIST = ("Дополнительное соглашение № D990000002-01\nк Договору № D990000001-01 от «21» января 2021 г.\n"
               "1. Внести в Договор следующие изменения:\n"
               "1.1. Пункт 3.1 Договора изложить в следующей редакции: «Ставка составляет 700 рублей».\n"
               "1.2. Пункт 5.2 Договора исключить.\n2. Настоящее Соглашение вступает в силу с 01.07.2026.")


def _ops_clauses(doc, supplement=True):
    return [c.clause_id for b in select_clauses(doc, "ops", load_settings(), supplement=supplement) for c in b]


def test_supplement_ops_routing_takes_the_change_list_and_its_sub_clauses():
    doc = text_doc([CHANGE_LIST])
    assert _ops_clauses(doc) == ["1", "1.1", "1.2", "2"]
    assert _ops_clauses(doc, supplement=False) == ["2"]  # main contracts keep their routing and recorded cache


def test_supplement_ops_routing_ignores_keywords_inside_quoted_new_wording():
    doc = text_doc([("Дополнительное соглашение № D990000002-01\nк Договору № D990000001-01 от «21» января 2021 г.\n"
                     "1. Раздел 9 Договора изложить в следующей редакции:\n«9. Расторжение Договора\n"
                     "9.1. Договор может быть расторгнут по соглашению Сторон.\n"
                     "9.2. Сторона вправе расторгнуть Договор в одностороннем порядке».\n"
                     "2. Настоящее Соглашение вступает в силу с 01.07.2026.\n"
                     "3. Соглашение составлено в двух экземплярах.")])
    assert _ops_clauses(doc) == ["1", "2"]


def test_supplement_with_few_clauses_goes_to_ops_whole():
    doc = text_doc([("Дополнительное соглашение № D990000002-01 к Договору № D990000001-01 от 21.01.2021\n"
                     "1. Стороны согласовали ставку 700 рублей с 01.07.2026.\n"
                     "2. Соглашение составлено в двух экземплярах.")])
    assert _ops_clauses(doc) == ["preamble", "1", "2"]


def test_auto_numbered_docx_supplement_reaches_ops(tmp_path):
    make_docx(tmp_path / "ds.docx", ["Дополнительное соглашение № D990000002-01 к Договору № D990000001-01 от 21.01.2021",
                                     "Пункт 3.1 Договора изложить в следующей редакции: ставка 700 рублей.",
                                     "Настоящее Соглашение вступает в силу с 01.07.2026."], numbered=(1, 2))
    (f,) = list_inputs(tmp_path)
    doc = read_document(f, FakeOcrEngine(), load_settings())
    assert [c.clause_id for c in doc.clauses] == ["preamble"] and _ops_clauses(doc) == ["preamble"]


def test_supplement_ops_call_sees_every_operation():
    client = ScriptedClient([_header_json(kind="supplement"), _dates_json(), '{"ops": []}'])
    understand_document(text_doc([CHANGE_LIST]), parse_header(CHANGE_LIST), client, load_settings(),
                        groups=["header", "ops"])
    body = client.calls[-1]["messages"][1]["content"]
    assert "1.1. Пункт 3.1" in body and "1.2. Пункт 5.2" in body


def test_ops_skipped_only_when_title_and_model_both_say_main():
    supplement = text_doc([("Дополнительное соглашение № D990000005-01\nг. Тверь 01.06.2026 г.\n"
                            "1. Внести в Договор аренды следующие изменения:\n1.1. Пункт 3.1 изложить в новой редакции.\n"
                            "2. Соглашение вступает в силу с 01.07.2026.")])
    client = ScriptedClient([_header_json(kind="main"), _dates_json(), '{"ops": []}'])
    understand_document(supplement, parse_header(supplement.pages[0].text), client, load_settings(),
                        groups=["header", "ops"])
    assert [c["schema"] for c in client.calls] == ["HeaderOut", "DatesOut", "OpsOut"]  # clause 2 states the term
    contract = text_doc([("Договор аренды № D990000006-01\nг. Тверь 01.06.2026 г.\n1. Предмет договора: аренда.\n"
                          "2. Договор может быть расторгнут по соглашению Сторон.")])
    client = ScriptedClient([_header_json(kind="main")])
    understand_document(contract, parse_header(contract.pages[0].text), client, load_settings(),
                        groups=["header", "ops"])
    assert [c["schema"] for c in client.calls] == ["HeaderOut"]


def test_quote_found_on_another_page():
    doc = text_doc(["1. Предмет договора", "Ставка вознаграждения составляет 700 рублей без НДС"])
    script = ['{"rates": [{"label": "Ставка", "amount": "700", "page": 1, "quote": "составляет 700 рублей без НДС"}]}']
    rec = understand_document(doc, HeaderGuess(), ScriptedClient(script), load_settings(), ["money"])
    assert rec.rates[0].status == "accepted" and rec.rates[0].source.page == 2


def _header_ids(doc):
    return [c.clause_id for b in select_clauses(doc, "header", load_settings()) for c in b]


TERM_LATE = ("Договор № D990000001-01\nг. Тверь 21.01.2021\n"
             + "\n".join(f"{i}. Условие номер {i} без дат." for i in range(1, 11))
             + "\n11. Настоящий Договор вступает в силу с 01.04.2021 и действует до 31.12.2026.\n12. Реквизиты сторон.")


def _term_ids(doc):
    return [c.clause_id for b in select_term_clauses(doc, load_settings()) for c in b]


def test_the_document_term_is_read_apart_from_the_header():
    doc = text_doc([TERM_LATE])
    assert _header_ids(doc) == ["preamble", "1", "2"] and _term_ids(doc) == ["11"]


def test_header_skips_a_term_that_is_only_quoted_new_wording():
    text = ("Дополнительное соглашение № D990000002-01\nк Договору № D990000001-01 от 21.01.2021\n"
            "1. Стороны договорились о следующем.\n2. Обязательства сторон сохраняются.\n"
            "3. Пункт 7.1 Договора изложить в следующей редакции: «7.1. Договор действует до 31.12.2027».\n"
            "4. Настоящее Соглашение вступает в силу с 01.01.2027.")
    assert _term_ids(text_doc([text])) == ["4"]


def test_a_later_header_batch_fills_a_field_the_first_did_not_find():
    long_subject = "1. Предмет договора: " + "оказание услуг по уборке помещений; " * 700
    doc = text_doc([(f"Договор № D990000001-01\nг. Тверь 21.01.2021\n{long_subject}\n2. Обязанности сторон.\n"
                     "9. Настоящий Договор вступает в силу с 01.04.2021.")])
    batches = len(select_clauses(doc, "header", load_settings()))
    assert batches > 1
    start = {"value": "2021-04-01", "page": 1, "quote": "Настоящий Договор вступает в силу с 01.04.2021"}
    script = [_header_json()] * (batches - 1) + [_header_json(effective_from=start), _dates_json()]
    rec = _understand(doc, script, groups=["header"])
    assert (rec.effective_from.status, rec.effective_from.value) == ("accepted", date(2021, 4, 1))


def _dates_json(**fields):
    return json.dumps({"effective_from": {"value": None}, "effective_to": {"value": None}, **fields}, ensure_ascii=False)


def test_dates_call_reads_a_late_term_clause_and_leaves_the_kind_to_the_header():
    start = {"value": "2021-04-01", "page": 1, "quote": "Настоящий Договор вступает в силу с 01.04.2021"}
    end = {"value": "2026-12-31", "page": 1, "quote": "и действует до 31.12.2026"}
    rec = _understand(text_doc([TERM_LATE]), [_header_json(kind="main"), _dates_json(effective_from=start, effective_to=end)],
                      groups=["header"])
    assert (rec.kind.value, rec.effective_from.value, rec.effective_to.value) == ("main", date(2021, 4, 1),
                                                                                  date(2026, 12, 31))


def test_entry_into_force_from_signing_leaves_the_start_to_the_signing_date():
    doc = text_doc([("Дополнительное соглашение № D990000002-01\nк Договору № D990000001-01 от 21.01.2021\n"
                     "1. Стороны договорились о следующем.\n2. Настоящее Соглашение вступает в силу с даты его подписания.")])
    filled_in = {"value": "2021-02-01", "page": 1, "quote": "Настоящее Соглашение вступает в силу с даты его подписания"}
    rec = _understand(doc, [_header_json(kind="supplement", effective_from=filled_in)], groups=["header"])
    assert (rec.effective_from.status, rec.effective_from.reason) == ("absent", "from_signing")

