import json
from datetime import date
from decimal import Decimal
from pathlib import Path

from contract_analyst.config import load_settings
from contract_analyst.llm.scripted import ScriptedClient, load_script
from contract_analyst.parsing.header import HeaderGuess, parse_header
from contract_analyst.understand.routing import estimate_tokens, select_clauses
from contract_analyst.understand.understand import understand_document
from tests.fixtures.builders import text_doc

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


def test_quote_found_on_another_page():
    doc = text_doc(["1. Предмет договора", "Ставка вознаграждения составляет 700 рублей без НДС"])
    script = ['{"rates": [{"label": "Ставка", "amount": "700", "page": 1, "quote": "составляет 700 рублей без НДС"}]}']
    rec = understand_document(doc, HeaderGuess(), ScriptedClient(script), load_settings(), ["money"])
    assert rec.rates[0].status == "accepted" and rec.rates[0].source.page == 2
