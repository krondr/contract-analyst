import json
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

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
    s = load_settings(llm_batch_tokens=500, llm_money_batch_tokens=500)
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



# --- the agreement an operation changes (fix/wrong-values-p4) -----------------------------------------------------

def _parent(number, quote):
    from contract_analyst.models import Extracted, ParentRef, Source

    return Extracted(value=ParentRef(number=number), status="accepted", source=Source(doc_id="d", page=1, quote=quote))


_DEFINES = text_doc([("Дополнительное соглашение № D990000009-01 к Договору № D990000001-01 от 19.08.2020 г.\n"
                      "Стороны заключили настоящее дополнительное соглашение (далее - «Соглашение») к договору "
                      "№ D990000001-01 от 19.08.2020 г. (далее - «Договор») и дополнительному соглашению "
                      "№ D990000005-01 от 29.08.2020 г. (далее – «Соглашение») о нижеследующем:\n"
                      "1. Изложить п. 1 – п. 14 Соглашения в следующей редакции.")])
_PLAIN = text_doc([("Дополнительное соглашение № D990000009-01 к Соглашению № D990000005-01 от 29.08.2020 г.\n"
                    "1. Изложить п. 1 – п. 14 Соглашения в следующей редакции.")])
_TO_CONTRACT = _parent("D990000001-01", "к Договору № D990000001-01 от 19.08.2020 г.")
_TO_AGREEMENT = _parent("D990000005-01", "к Соглашению № D990000005-01 от 29.08.2020 г.")


@pytest.mark.parametrize("target, doc, parent, expected", [
    ("п. 1–14 Соглашения", _PLAIN, _TO_AGREEMENT, "п. 1–14 Соглашения № D990000005-01"),
    ("п. 1–14 Соглашения", _DEFINES, _TO_CONTRACT, "п. 1–14 Соглашения № D990000005-01"),
    ("п. 3 настоящего Соглашения", _PLAIN, _TO_AGREEMENT, "п. 3 настоящего Соглашения"),
    ("п. 5 Договора", _PLAIN, _TO_AGREEMENT, "п. 5 Договора"),
    ("п. 2 Соглашения № D990000007-01", _PLAIN, _TO_AGREEMENT, "п. 2 Соглашения № D990000007-01"),
    ("п. 1–14 Соглашения", _PLAIN, _TO_CONTRACT, "п. 1–14 Соглашения"),
], ids=["parent_is_the_agreement", "agreement_defined_in_the_text", "this_agreement", "the_contract", "numbered",
        "parent_is_the_contract"])
def test_an_operation_on_an_agreement_names_which_one(target, doc, parent, expected):
    """«п. 1–14 Соглашения» with no number would be applied to the contract itself. The number is the one the text
    defines as «Соглашение», or the parent's when the document calls its parent an agreement; a parent that is the
    contract does not name the agreement."""
    from contract_analyst.understand.understand import name_amended_agreement

    assert name_amended_agreement(target, doc, parent) == expected


def test_the_header_batch_reads_the_clause_where_the_city_and_date_stand():
    """A contract that opens with «1. Определения»: the line «г. Москва «25» января 2021 г.» comes after the
    definitions, and segmentation files it under the last definition."""
    doc = text_doc([("ДОГОВОР № D990000001-01\n1. Определения\n1.1. Абонент – лицо, заключившее договор.\n"
                     "1.2. Тариф – тарифный план.\n1.6. Комплект – SIM-карта.\n1.6.1. Стартовый комплект.\n"
                     "1.6.2. Набор услуг.\nг. Москва «25» января 2021 г.\nАО «Альфа-Связь», в лице директора, с одной "
                     "стороны, и ООО «Ромашка», с другой стороны, заключили настоящий Договор.\n"
                     "2. Предмет договора\n2.1. Партнер оказывает услуги.")])
    picked = [c.clause_id for b in select_clauses(doc, "header", load_settings()) for c in b]
    assert "1.6.2" in picked and "2.1" not in picked and "1.6.1" not in picked


def test_money_clauses_are_packed_into_their_own_smaller_batches():
    """A rate table answer carries a quote per row: the money group packs fewer clauses per call, the others do not."""
    s = load_settings(llm_batch_tokens=6000, llm_money_batch_tokens=1000)
    doc = text_doc([f"{i}. Ставка {i} составляет {i * 10} рублей, срок оплаты 5 дней. " + "слово " * 400
                    for i in range(1, 21)])
    assert all(estimate_tokens(b) <= 1000 for b in select_clauses(doc, "money", s))
    assert max(estimate_tokens(b) for b in select_clauses(doc, "core_terms", s)) > 1000


@pytest.mark.parametrize("op, target, parent, expected", [
    ("replace_clause", "п. 5.1 Договора № D990000001-01", _TO_CONTRACT, "п. 5.1 Договора"),
    ("replace_appendix", "Приложение № 2 к Договору № D990000001-01", _TO_CONTRACT, "Приложение № 2 к Договору"),
    ("terminate", "Договор № D990000001-01", _TO_CONTRACT, "Договор"),
    ("replace_clause", "Договор № D990000001-01", _TO_CONTRACT, "Договор № D990000001-01"),
    ("replace_clause", "п. 2 Соглашения № D990000005-01", _TO_CONTRACT, "п. 2 Соглашения № D990000005-01"),
    ("replace_clause", "п. 2 Соглашения № D990000005-01", _TO_AGREEMENT, "п. 2 Соглашения № D990000005-01"),
    ("invalidate_document", "D990000001-01", _TO_CONTRACT, "D990000001-01"),
], ids=["clause_of_the_contract", "appendix_of_the_contract", "termination_of_the_contract", "nothing_else_left",
        "another_agreement", "parent_is_the_agreement", "invalidation"])
def test_the_number_of_the_contract_a_supplement_amends_is_dropped_from_its_targets(op, target, parent, expected):
    """A supplement changes its contract unless it names another document: «п. 5.1 Договора № <its contract>» is
    «п. 5.1 Договора». The number stays where it names another document, where nothing else would be left of a
    clause operation's target, and in an invalidation, whose target is the number."""
    from contract_analyst.understand.understand import drop_own_contract_number

    assert drop_own_contract_number(op, target, parent) == expected


def test_a_start_the_gate_refused_is_replaced_by_the_term_call():
    """The header batch offered the signing date as the start and the gate refused it (no «вступает в силу» there);
    the term clause states the real start, and its answer must not be thrown away for the refused one."""
    signing = {"value": "2021-01-21", "page": 1, "quote": "г. Тверь 21.01.2021"}
    start = {"value": "2021-04-01", "page": 1, "quote": "Настоящий Договор вступает в силу с 01.04.2021"}
    end = {"value": "2026-12-31", "page": 1, "quote": "и действует до 31.12.2026"}
    rec = _understand(text_doc([TERM_LATE]), [_header_json(kind="main", effective_from=signing),
                                              _dates_json(effective_from=start, effective_to=end)], groups=["header"])
    assert (rec.effective_from.status, rec.effective_from.value) == ("accepted", date(2021, 4, 1))


def test_a_refused_start_stays_refused_when_the_term_call_finds_nothing_better():
    signing = {"value": "2021-01-21", "page": 1, "quote": "г. Тверь 21.01.2021"}
    rec = _understand(text_doc([TERM_LATE]), [_header_json(kind="main", effective_from=signing), _dates_json()],
                      groups=["header"])
    assert rec.effective_from.status == "rejected"


# --- the start of a document that does not state one (fix/wrong-values-p8) -----------------------------------------

_NO_START = ("Дополнительное соглашение № D990000002-01\nк Договору № D990000001-01 от 21.01.2021\nг. Тверь 01.02.2021\n"
             "1. Стоимость услуги составляет 500 рублей.\n2. Стоимость выезда составляет 900 рублей с 01.03.2021.\n"
             "3. Оплата производится в течение 5 дней с даты подписания Акта.")
_RATES = json.dumps({"rates": [
    {"label": "Стоимость услуги", "amount": "500", "page": 1, "quote": "Стоимость услуги составляет 500 рублей"},
    {"label": "Стоимость выезда", "amount": "900", "valid_from": "2021-03-01", "page": 1,
     "quote": "Стоимость выезда составляет 900 рублей с 01.03.2021"}]}, ensure_ascii=False)


def test_a_document_that_does_not_say_when_it_starts_has_no_start_and_its_undated_rates_wait():
    """The signing date is the start only where the text says so: «с даты подписания Акта» is about a payment.
    A rate without a start of its own is shown but not accepted; a rate with its own start stays."""
    rec = _understand(text_doc([_NO_START]), [_header_json(kind="supplement"), _RATES], groups=["header", "money"])
    assert (rec.effective_from.status, rec.effective_from.reason) == ("absent", "start_not_stated")
    statuses = {r.amount: (r.status, (r.reason or "").split(":")[0]) for r in rec.rates}
    assert statuses == {Decimal(500): ("conflict", "start_unknown"), Decimal(900): ("accepted", "")}


@pytest.mark.parametrize("clause", ["4. Настоящее Соглашение вступает в силу со дня его подписания Сторонами.",
                                    "4. Соглашение действует с момента заключения.",
                                    "4. Настоящее Соглашение вступает в силу после его подписания."])
def test_a_document_that_starts_from_signing_keeps_the_signing_date_as_its_start(clause):
    rec = _understand(text_doc([f"{_NO_START}\n{clause}"]), [_header_json(kind="supplement"), _dates_json(), _RATES],
                      groups=["header", "money"])
    assert (rec.effective_from.status, rec.effective_from.reason) == ("absent", "from_signing")
    assert all(r.status == "accepted" for r in rec.rates)


def test_the_start_is_not_judged_when_the_header_was_not_read():
    rec = _understand(text_doc([_NO_START]), [_RATES], groups=["money"])
    assert all(r.status == "accepted" for r in rec.rates)


def test_the_fold_dates_a_document_from_its_signing_only_when_the_text_says_so():
    from contract_analyst.assemble.consolidate import document_start

    rec = _understand(text_doc([_NO_START]), [_header_json(kind="supplement"), _RATES], groups=["header", "money"])
    signed = rec.model_copy(update={"effective_from": rec.effective_from.model_copy(update={"reason": "from_signing"})})
    assert rec.doc_date.status == "accepted"
    assert (document_start(rec), document_start(signed)) == (None, rec.doc_date.value)


@pytest.mark.parametrize("clause", [
    "Настоящее Соглашение применяется к отношениям Сторон с 01.03.2021.",
    "Настоящее Соглашение распространяет свое действие на отношения Сторон, возникшие с 01.03.2021.",
    "Условия настоящего Дополнительного соглашения вводятся с 01.03.2021."])
def test_the_term_call_reads_a_clause_that_says_from_when_the_document_applies(clause):
    text = ("Дополнительное соглашение № D990000002-01\nг. Тверь 21.01.2021\n"
            + "\n".join(f"{i}. Условие номер {i} без дат." for i in range(1, 6)) + f"\n6. {clause}")
    assert _term_ids(text_doc([text])) == ["6"]


def test_the_term_call_skips_a_clause_that_applies_a_rate_from_a_date():
    text = ("Дополнительное соглашение № D990000002-01\nг. Тверь 21.01.2021\n"
            + "\n".join(f"{i}. Условие номер {i} без дат." for i in range(1, 6))
            + "\n6. Ставка вознаграждения применяется с 01.03.2021.")
    assert _term_ids(text_doc([text])) == []


@pytest.mark.parametrize("clause", [
    "Правила о договорной неустойке применяются при условии направления Акта о нарушениях.",
    "Положения настоящего Приложения, касающиеся проверки договоров, распространяются на все Абонентские договоры.",
    "На условиях настоящего Договора Исполнитель применяет тарифы Заказчика."])
def test_the_term_call_skips_a_clause_where_something_else_applies(clause):
    """Only the document itself applying from a date is its term; a rule, a provision or a tariff that applies is
    not, and a long contract has dozens of such clauses that would drown the real one."""
    text = ("Дополнительное соглашение № D990000002-01\nг. Тверь 21.01.2021\n"
            + "\n".join(f"{i}. Условие номер {i} без дат." for i in range(1, 6)) + f"\n6. {clause}")
    assert _term_ids(text_doc([text])) == []


# --- review fixes (fix/review-p9) ------------------------------------------------------------------------------------

_LONG_OPS = ("Дополнительное соглашение № D990000002-01\nк Договору № D990000001-01 от 21.01.2021\n"
             "1. Внести в Договор следующие изменения:\n"
             "1.1. Пункт 3.1 Договора изложить в следующей редакции: «Ставка составляет 700 рублей». "
             + "слово " * 300 + "\n"
             "1.2. Пункт 5.2 Договора исключить. " + "слово " * 300 + "\n"
             "2. Настоящее Соглашение вступает в силу с 01.07.2026.")


def test_a_start_a_later_ops_batch_grounds_wins_over_one_the_gate_refused_in_an_earlier_batch():
    s = load_settings(llm_batch_tokens=500)
    doc = text_doc([_LONG_OPS])
    batches = select_clauses(doc, "ops", s, supplement=True)
    assert len(batches) > 1
    wrong = {"value": "2026-08-01", "page": 1, "quote": "Соглашение вступает в силу с 01.08.2026"}
    right = {"value": "2026-07-01", "page": 1, "quote": "Настоящее Соглашение вступает в силу с 01.07.2026"}
    script = ([json.dumps({"ops": [], "effective_from": wrong}, ensure_ascii=False)]
              + [json.dumps({"ops": []})] * (len(batches) - 2)
              + [json.dumps({"ops": [], "effective_from": right}, ensure_ascii=False)])
    rec = understand_document(doc, parse_header(_LONG_OPS), ScriptedClient(script), s, ["ops"])
    assert (rec.effective_from.status, rec.effective_from.value) == ("accepted", date(2026, 7, 1))


def test_a_start_the_gate_refused_in_the_ops_call_does_not_replace_a_start_not_stated():
    """A date the gate refused is no evidence: the document still does not say when it starts."""
    wrong = {"value": "2021-05-01", "page": 1, "quote": "вступает в силу с 01.05.2021"}
    ops = json.dumps({"ops": [], "effective_from": wrong}, ensure_ascii=False)
    text = CHANGE_LIST.split("\n2. ")[0]  # the change list without the clause that says when it starts
    client = ScriptedClient([_header_json(kind="supplement"), ops])
    rec = understand_document(text_doc([text]), parse_header(text), client, load_settings(), ["header", "ops"])
    assert [c["schema"] for c in client.calls] == ["HeaderOut", "OpsOut"]
    assert (rec.effective_from.status, rec.effective_from.reason) == ("absent", "start_not_stated")


def _held(rec):
    return {r.amount: r.status for r in rec.rates}


def test_rates_read_after_the_header_wait_for_a_start_the_document_does_not_state():
    """The agent may read the header first and the money later: the rates still wait for a start."""
    from contract_analyst.agent.steps import _merge_records

    header = _understand(text_doc([_NO_START]), [_header_json(kind="supplement")], groups=["header"])
    money = _understand(text_doc([_NO_START]), [_RATES], groups=["money"])
    assert _held(_merge_records(header, money, ["money"])) == {Decimal(500): "conflict", Decimal(900): "accepted"}
    assert _held(_merge_records(money, header, ["header"])) == {Decimal(500): "conflict", Decimal(900): "accepted"}


def test_rates_held_for_a_start_are_released_when_a_header_reread_finds_one():
    from contract_analyst.agent.steps import _merge_records

    held = _understand(text_doc([_NO_START]), [_header_json(kind="supplement"), _RATES], groups=["header", "money"])
    signed = f"{_NO_START}\n4. Настоящее Соглашение вступает в силу со дня его подписания Сторонами."
    header = _understand(text_doc([signed]), [_header_json(kind="supplement"), _dates_json()], groups=["header"])
    merged = _merge_records(held, header, ["header"])
    assert merged.effective_from.reason == "from_signing"
    assert _held(merged) == {Decimal(500): "accepted", Decimal(900): "accepted"}


def test_a_reread_group_drops_its_old_model_failure_and_issues_are_not_repeated():
    """Once the ops group answers on a re-read, its old timeout no longer marks the operations unread."""
    from contract_analyst.agent.steps import _merge_records
    from contract_analyst.models import Issue

    timeout = Issue(code="llm_timeout", message="ДС.pdf: группа ops — LLM не ответила: timeout")
    money_failed = Issue(code="llm_truncated", message="ДС.pdf: группа money — ответ оборвался")
    draft = Issue(code="draft", message="ДС.pdf: похоже на проект, подписание не подтверждено")
    rec = _understand(text_doc([_NO_START]), [_header_json(kind="supplement")], groups=["header"])
    old = rec.model_copy(update={"issues": [timeout, money_failed, draft]})
    new = rec.model_copy(update={"issues": [draft]})
    assert [i.code for i in _merge_records(old, new, ["ops"]).issues] == ["llm_truncated", "draft"]
