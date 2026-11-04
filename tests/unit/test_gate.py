from datetime import date
from decimal import Decimal

import pytest

from contract_analyst.config import load_settings
from contract_analyst.gate.checks import (
    check_format,
    check_ocr_confidence,
    check_quote_present,
    check_value_in_quote,
)
from contract_analyst.gate.gate import Candidate, apply_gate, gate_op, gate_rate
from contract_analyst.models import OcrWord, Page, PageQuality
from contract_analyst.understand.schemas import OpOut, RateOut
from tests.fixtures.builders import ocr_doc, text_doc

TEXT_PAGE = Page(number=1, method="text", text="Ставка составляет 1500 рублей. Договор от «21» января 2021 г.",
                 quality=PageQuality(chars=60))


@pytest.fixture
def doc():
    return text_doc([("Ставка составляет 1500 рублей. Договор от «21» января 2021 г.\n"
                     "Тариф А: 700 рублей без НДС с 01.07.2026.\n"
                     "Соглашение № D990000003-01 от 14.02.2024 считать недействительным.")])


def test_quote_verbatim_on_text_page():
    assert check_quote_present("1500 рублей", TEXT_PAGE, 90).ok
    assert not check_quote_present("2500 рублей", TEXT_PAGE, 90).ok


def test_quote_ignores_typographic_quotes_and_line_breaks():
    assert check_quote_present('Договор от "21" января\n2021 г.', TEXT_PAGE, 90).ok


def test_ocr_page_allows_fuzzy_quote():
    p = TEXT_PAGE.model_copy(update={"method": "ocr", "text": "Ставка составпяет 1500 рублей."})
    assert check_quote_present("Ставка составляет 1500 рублей", p, 90).ok
    assert not check_quote_present("Срок действия до 2030 года", p, 90).ok


def test_value_must_follow_from_quote():
    assert check_value_in_quote("2021-01-21", "от «21» января 2021 г.", "date").ok
    assert not check_value_in_quote("2021-01-25", "от «21» января 2021 г.", "date").ok
    assert check_value_in_quote("700", "ставка 700,0 рублей", "money").ok
    assert not check_value_in_quote("D990000001-01", "к Договору № D990000002-01", "doc_number").ok


def test_formats():
    assert not check_format("D12345", "doc_number").ok
    assert check_format("D990000001-01", "doc_number").ok
    assert not check_format("1990-01-01", "date").ok
    assert not check_format("-5", "money").ok


def test_low_ocr_conf_rejected():
    p = TEXT_PAGE.model_copy(update={"method": "ocr", "words": [OcrWord(text="1500", conf=0.5)]})
    assert not check_ocr_confidence("1500", p, 0.8).ok
    assert check_ocr_confidence("1500", p.model_copy(update={"words": [OcrWord(text="1500", conf=0.9)]}), 0.8).ok


def test_statuses(doc):
    s = load_settings()
    assert apply_gate(Candidate(field="name", value="Ромашка", kind="text", page=1, quote="- - - -"), doc, s).status == "redacted"
    assert apply_gate(Candidate(field="name", value=None, kind="text", page=1, quote="«  »", redacted=True), doc, s).status == "redacted"
    r = apply_gate(Candidate(field="amount", value="2500", kind="money", page=1, quote="2500 рублей"), doc, s)
    assert r.status == "rejected" and r.reason.startswith("quote_not_found")
    assert apply_gate(Candidate(field="x", value=None, kind="text", page=None, quote=None), doc, s).status == "absent"
    ok = apply_gate(Candidate(field="doc_date", value="2021-01-21", kind="date", page=1, quote="от «21» января 2021 г."), doc, s)
    assert ok.status == "accepted" and ok.source.doc_id == doc.doc_id and ok.source.page == 1


def test_gate_rate_requires_amount_in_quote(doc):
    s = load_settings()
    good = gate_rate(RateOut(label="Тариф А", amount="700", vat="excluded", valid_from="2026-07-01", page=1,
                             quote="Тариф А: 700 рублей без НДС"), doc, s)
    assert good.status == "accepted" and good.amount == Decimal(700) and good.valid_from == date(2026, 7, 1)
    bad = gate_rate(RateOut(label="Тариф А", amount="900", page=1, quote="Тариф А: 700 рублей без НДС"), doc, s)
    assert bad.status == "rejected"


def test_gate_op_invalidation_target_must_be_quoted(doc):
    s = load_settings()
    quote = "Соглашение № D990000003-01 от 14.02.2024 считать недействительным"
    assert gate_op(OpOut(op="invalidate_document", target="D990000003-01", page=1, quote=quote), doc, s).status == "accepted"
    assert gate_op(OpOut(op="invalidate_document", target="D990000009-01", page=1, quote=quote), doc, s).status == "rejected"


@pytest.mark.parametrize(("op", "text"), [
    ("terminate", "1. Расторгнуть Дополнительное соглашение № D990000003-01 с 01.05.2026."),
    ("suspend_part", "1. Прекратить с 01.05.2026 привлечение абонентов по Дополнительному соглашению № D990000003-01.")])
def test_ending_operation_target_number_must_be_quoted(op, text):
    quote = text.split(". ", 1)[1].rstrip(".")

    def ending(target):
        return gate_op(OpOut(op=op, target=target, effective_from="2026-05-01", page=1, quote=quote), text_doc([text]),
                       load_settings())

    assert ending("D990000003-01").status == "accepted"
    wrong = ending("D990000004-01")
    assert wrong.status == "rejected" and wrong.reason.startswith("value_not_in_quote")
    assert ending("Дополнительное соглашение").status == "accepted"


def test_rate_quote_reanchored_to_the_line_with_label_and_amount(doc):
    row = gate_rate(RateOut(label="Тариф А", amount="700", page=1, quote="Тариф А"), doc, load_settings())
    assert row.status == "accepted" and "700" in row.source.quote and row.reason == "quote_reanchored"


def test_rate_not_reanchored_when_no_line_has_both(doc):
    row = gate_rate(RateOut(label="Тариф Б", amount="700", page=1, quote="Тариф Б"), doc, load_settings())
    assert row.status == "rejected"


def test_rate_reanchored_in_table_with_cells_on_separate_lines():
    table = text_doc(["Таблица №1\nСтавка (руб. без НДС) с 01.07.2026\nс 01.01.2027\nБазовый\n280\n520\nОптима\n340,5\n520\n"])
    row = gate_rate(RateOut(label="Оптима", amount="520", valid_from="2027-01-01", page=1,
                            quote="Ставка (руб. без НДС) с 01.07.2026"), table, load_settings())
    assert row.status == "accepted" and row.source.quote == "Оптима 340,5 520"
    wrong = gate_rate(RateOut(label="Оптима", amount="280", page=1, quote="Ставка"), table, load_settings())
    assert wrong.status == "rejected"


def test_accepted_value_source_names_its_clause():
    doc = text_doc([("12.2. Уведомления направляются по факсу.\nПриложение № 1\nк Договору\n"
                     "1.6. Вознаграждение составляет 60% от начислений.")])
    s = load_settings()
    notice = apply_gate(Candidate(field="notification_channel", value="по факсу", kind="text", page=1,
                                  quote="Уведомления направляются по факсу."), doc, s)
    fee = apply_gate(Candidate(field="payment_terms", value="60% от начислений", kind="text", page=1,
                               quote="Вознаграждение составляет 60% от начислений"), doc, s)
    assert (notice.source.clause_id, fee.source.clause_id) == ("12.2", "Приложение 1/1.6")


def test_value_without_quote_is_rejected_not_a_crash(doc):
    for kind, value in [("text", "длинное значение условия"), ("date", "2021-01-21"), ("money", "1500")]:
        r = apply_gate(Candidate(field="x", value=value, kind=kind, page=1, quote=None), doc, load_settings())
        assert r.status == "rejected" and r.reason.startswith("no_quote")


def test_ocr_value_is_checked_against_the_page_not_the_fuzzy_quote():
    doc = ocr_doc("Ставка вознаграждения составляет 1600 рублей без НДС за каждого абонента.")
    quote = "Ставка вознаграждения составляет 1500 рублей без НДС за каждого абонента."
    s = load_settings()
    r = apply_gate(Candidate(field="x", value="1500", kind="money", page=1, quote=quote), doc, s)
    assert r.status == "rejected" and r.reason.startswith("value_not_on_page")
    assert gate_rate(RateOut(label="Ставка вознаграждения", amount="1500", page=1, quote=quote), doc, s).status == "rejected"
    fixed = quote.replace("1500", "1600")
    assert apply_gate(Candidate(field="x", value="1600", kind="money", page=1, quote=fixed), doc, s).status == "accepted"


def test_ocr_value_word_itself_must_be_confident():
    doc = ocr_doc("Ставка 1500 рублей.\nПени 1500 рублей.", conf={1: 0.5})  # only the first 1500 is unsure
    s = load_settings()
    r = apply_gate(Candidate(field="x", value="1500", kind="money", page=1, quote="Ставка 1500 рублей."), doc, s)
    assert r.status == "rejected" and r.reason.startswith("ocr_value_unconfirmed")
    assert apply_gate(Candidate(field="x", value="1500", kind="money", page=1, quote="Пени 1500 рублей."), doc,
                      s).status == "accepted"


def test_ocr_value_word_missing_from_recognized_words_fails_closed():
    doc = ocr_doc("Ставка 1500 рублей.", words=["Ставка", "рублей."])
    r = apply_gate(Candidate(field="x", value="1500", kind="money", page=1, quote="Ставка 1500 рублей."), doc,
                   load_settings())
    assert r.status == "rejected"


@pytest.mark.parametrize("title", ["Дополнительное соглашение № Д990000015-01",
                                   "Дополнительное соглашение № 0990000015-01",
                                   "Дополнительное соглашение №Д990000015-01"])
def test_scanned_document_number_with_misread_d_is_confirmed(title):
    s = load_settings()
    c = Candidate(field="number", value="D990000015-01", kind="doc_number", page=1, quote=title)
    assert apply_gate(c, ocr_doc(title), s).status == "accepted"
    assert apply_gate(c, ocr_doc(title, conf={len(title.split()) - 1: 0.5}), s).status == "rejected"


def test_split_reading_of_a_grouped_amount_is_rejected():
    quote = "Вознаграждение составляет 1 500 рублей за абонента."
    doc, s = text_doc([quote]), load_settings()
    for value in ("500", "1"):
        assert apply_gate(Candidate(field="x", value=value, kind="money", page=1, quote=quote), doc, s).status == "rejected"
        assert gate_rate(RateOut(label="Вознаграждение", amount=value, page=1, quote=quote), doc, s).status == "rejected"
    assert gate_rate(RateOut(label="Вознаграждение", amount="1 500", page=1, quote=quote), doc, s).status == "accepted"


def test_quote_starting_inside_a_number_is_not_evidence():
    doc = text_doc(["Цена товара составляет 21500 рублей."])
    r = apply_gate(Candidate(field="x", value="1500", kind="money", page=1, quote="1500 рублей"), doc, load_settings())
    assert r.status == "rejected" and r.reason.startswith("value_not_on_page")


def test_numbers_and_dates_of_a_text_value_must_be_on_the_page():
    doc = text_doc([("8.2. За нарушение срока оплаты Арендатор уплачивает штраф 1000 рублей за каждый день "
                     "просрочки.\n9.1. Договор действует до «31» декабря 2026 года.")])
    s = load_settings()

    def term(value, quote):
        return apply_gate(Candidate(field="t", value=value, kind="text", page=1, quote=quote), doc, s).status

    penalty = "Арендатор уплачивает штраф 1000 рублей за каждый день просрочки"
    assert term("штраф 5000 рублей за каждый день просрочки", penalty) == "rejected"
    assert term("штраф 1000 рублей за каждый день просрочки", penalty) == "accepted"
    validity = "Договор действует до «31» декабря 2026 года"
    assert term("действует до 31.12.2026", validity) == "accepted"
    assert term("действует до 2026-12-31", validity) == "accepted"  # the model may write dates in ISO form
    assert term("действует до 31.12.2027", validity) == "rejected"


def test_amount_read_from_table_cells_on_one_line_is_conflict():
    doc = text_doc(["Ставка, руб. без НДС с 01.07.2026 с 01.01.2027\nБазовый 280 520\nОптима 340,5 520"])
    s = load_settings()
    r = gate_rate(RateOut(label="Базовый", amount="520", valid_from="2027-01-01", page=1, quote="Базовый 280 520"), doc, s)
    assert r.status == "conflict" and r.reason.startswith("ambiguous_number")
    ok = gate_rate(RateOut(label="Оптима", amount="340,5", valid_from="2026-07-01", page=1,
                           quote="Оптима 340,5 520"), doc, s)
    assert ok.status == "accepted"


def test_rate_reanchoring_stays_on_the_hinted_page():
    doc = text_doc(["Ставки приведены в таблице на следующей странице.", "Тариф А 700"])
    r = gate_rate(RateOut(label="Тариф А", amount="700", page=1, quote="Ставки приведены в таблице"), doc,
                  load_settings())
    assert r.status == "rejected"


def test_rate_period_must_be_stated_on_its_row_or_in_the_header_above():
    doc = text_doc([("Приложение № 7\nСтавки на период с 01.07.2026 по 31.12.2026:\nТариф А 500\n"
                     "Ставки с 01.01.2027:\nТариф А 800")])
    s = load_settings()

    def rate(amount, start, end, quote="Тариф А"):
        return gate_rate(RateOut(label="Тариф А", amount=amount, valid_from=start, valid_to=end, page=1,
                                 quote=f"{quote} {amount}" if quote == "Тариф А" else quote), doc, s)

    swapped = rate("800", "2026-07-01", "2026-12-31")
    assert swapped.status == "conflict" and swapped.reason.startswith("date_not_near")
    assert rate("500", "2027-01-01", None).status == "conflict"
    assert rate("500", "2026-07-01", "2026-12-31").status == "accepted"
    assert rate("800", "2027-01-01", None).status == "accepted"
    both = "Ставки на период с 01.07.2026 по 31.12.2026: Тариф А 500 Ставки с 01.01.2027: Тариф А 800"
    assert rate("800", "2026-07-01", "2026-12-31", quote=both).status == "conflict"
    assert rate("500", "2026-07-01", "2026-12-31", quote=both).status == "accepted"


def test_rate_period_later_in_the_same_sentence_counts():
    doc = text_doc([("2. Ставка вознаграждения составляет 500 рублей без НДС\nи применяется с 01.07.2026 по "
                     "31.12.2026.\n3. С 01.01.2027 ставка вознаграждения составляет 800 рублей без НДС.")])
    s = load_settings()

    def rate(amount, start, end):
        return gate_rate(RateOut(label="Ставка вознаграждения", amount=amount, valid_from=start, valid_to=end, page=1,
                                 quote=f"Ставка вознаграждения составляет {amount} рублей без НДС"), doc, s).status

    assert rate("500", "2026-07-01", "2026-12-31") == "accepted"
    assert rate("500", "2027-01-01", None) == "conflict"
    assert rate("800", "2027-01-01", None) == "accepted"


def test_op_date_comes_from_its_own_sentence_or_the_entry_into_force():
    s = load_settings()
    dated = text_doc([("1. Расторгнуть Договор № D990000001-01 от 21.01.2021 с 01.01.2027.\n"
                       "2. Настоящее Соглашение вступает в силу с 01.07.2026.")])

    def terminate(doc, start, quote):
        return gate_op(OpOut(op="terminate", target="D990000001-01", effective_from=start, page=1, quote=quote),
                       doc, s)

    quote = "Расторгнуть Договор № D990000001-01 от 21.01.2021 с 01.01.2027"
    assert terminate(dated, "2027-01-01", quote).status == "accepted"
    wrong = terminate(dated, "2026-07-01", quote)
    assert wrong.status == "conflict" and wrong.reason.startswith("date_not_near")
    assert terminate(dated, "2021-01-21", quote).status == "conflict"  # the contract's own date is a reference
    undated = text_doc([("1. Расторгнуть Договор № D990000001-01 от 21.01.2021.\n"
                         "2. Настоящее Соглашение вступает в силу с 01.07.2026.")])
    quote = "Расторгнуть Договор № D990000001-01 от 21.01.2021"
    assert terminate(undated, "2026-07-01", quote).status == "accepted"
    assert terminate(undated, "2021-01-21", quote).status == "conflict"


@pytest.mark.parametrize("op,text,quote", [
    ("terminate", "4.3. Договор может быть расторгнут по соглашению Сторон с 01.07.2026.",
     "Договор может быть расторгнут по соглашению Сторон"),
    ("terminate", "4.3. Договор может быть расторгнут по соглашению Сторон с 01.07.2026.",
     "расторгнут по соглашению Сторон с 01.07.2026"),
    ("terminate", "4.4. Арендодатель вправе расторгнуть Договор с 01.07.2026 в одностороннем порядке.",
     "Арендодатель вправе расторгнуть Договор с 01.07.2026"),
    ("terminate", "4.5. В случае просрочки оплаты Договор расторгается с 01.07.2026.",
     "Договор расторгается с 01.07.2026"),
    ("terminate", "1. Договор не расторгается и действует до 01.07.2026.", "Договор не расторгается"),
    ("suspend_part", "2. Компания вправе прекратить привлечение новых абонентов с 01.07.2026.",
     "прекратить привлечение новых абонентов с 01.07.2026"),
    ("restate_document", "1. Договор (в редакции Дополнительного соглашения № 1 от 01.07.2026) действует.",
     "Договор (в редакции Дополнительного соглашения № 1 от 01.07.2026)")])
def test_high_stakes_op_without_operative_wording_is_rejected(op, text, quote):
    r = gate_op(OpOut(op=op, effective_from="2026-07-01", page=1, quote=quote), text_doc([text]), load_settings())
    assert r.status == "rejected" and r.reason.startswith("not_operative")


@pytest.mark.parametrize("op,text", [
    ("terminate", "1. Стороны договорились расторгнуть Договор с 01.07.2026."),
    ("terminate", "1. Договор считается расторгнутым с 01.07.2026."),
    ("terminate", "1. Стороны пришли к соглашению о расторжении Договора с 01.07.2026."),
    ("suspend_part", "1. Прекратить с 01.07.2026 привлечение новых абонентов по Договору."),
    ("restate_document", "1. Изложить Договор в новой редакции с 01.07.2026 согласно Приложению № 1.")])
def test_high_stakes_op_with_operative_wording_is_accepted(op, text):
    quote = text.split(". ", 1)[1].rstrip(".")
    assert gate_op(OpOut(op=op, effective_from="2026-07-01", page=1, quote=quote), text_doc([text]),
                   load_settings()).status == "accepted"


def test_terminate_without_a_date_is_conflict():
    doc = text_doc(["1. Стороны договорились расторгнуть Договор."])
    r = gate_op(OpOut(op="terminate", page=1, quote="Стороны договорились расторгнуть Договор"), doc, load_settings())
    assert r.status == "conflict" and r.reason.startswith("terminate_without_date")


def test_rate_reanchoring_with_several_candidates_is_conflict():
    doc = text_doc(["Ставки на 2026 год:\nТариф А 700\nСтавки на 2027 год:\nТариф А: 700 рублей"])
    r = gate_rate(RateOut(label="Тариф А", amount="700", page=1, quote="Ставки на 2026 год"), doc, load_settings())
    assert r.status == "conflict" and r.reason.startswith("reanchor_ambiguous")


def test_effective_date_must_be_a_start_not_an_end():
    quote = "Настоящее Соглашение вступает в силу с 01.01.2017 и действует до 31.12.2017"
    doc, s = text_doc([quote + "."]), load_settings()

    def eff(value):
        return apply_gate(Candidate(field="effective_from", value=value, kind="date", page=1, quote=quote), doc, s)

    assert eff("2017-01-01").status == "accepted"
    wrong = eff("2017-12-31")
    assert wrong.status == "rejected" and wrong.reason.startswith("date_role")


def test_document_date_cannot_be_the_date_of_the_contract_it_refers_to():
    s = load_settings()
    doc = text_doc([("Дополнительное соглашение № D990000002-01\nк Договору № D990000001-01 от 02.06.2023\n"
                     "г. Тверь 15.03.2024")])

    def signed(value, quote, d=doc):
        return apply_gate(Candidate(field="doc_date", value=value, kind="date", page=1, quote=quote), d, s)

    ref = signed("2023-06-02", "к Договору № D990000001-01 от 02.06.2023")
    assert ref.status == "rejected" and ref.reason.startswith("date_role")
    assert signed("2024-03-15", "г. Тверь 15.03.2024").status == "accepted"
    title = text_doc(["Договор № D990000001-01 от 02.06.2023"])
    assert signed("2023-06-02", "Договор № D990000001-01 от 02.06.2023", title).status == "accepted"
    both = "Дополнительное соглашение № D990000002-01 от 15.03.2024 к Договору № D990000001-01 от 02.06.2023"
    two = text_doc([both])
    assert signed("2024-03-15", both, two).status == "accepted"
    assert signed("2023-06-02", both, two).status == "rejected"


def test_effective_date_taken_from_a_reference_is_rejected():
    doc = text_doc(["Изменения к Договору № D990000001-01 от 02.06.2023 вносятся настоящим Соглашением."])
    r = apply_gate(Candidate(field="effective_from", value="2023-06-02", kind="date", page=1,
                             quote="к Договору № D990000001-01 от 02.06.2023"), doc, load_settings())
    assert r.status == "rejected" and r.reason.startswith("date_role")


@pytest.mark.parametrize("op", ["replace_clause", "delete_clause", "replace_appendix", "restate_section", "add_clause"])
def test_clause_operation_without_a_target_is_rejected(op):
    text = "1. Изложить в новой редакции с 01.07.2026 следующим образом: «Арендная плата составляет 120 000 рублей»."
    r = gate_op(OpOut(op=op, effective_from="2026-07-01", page=1, quote="Изложить в новой редакции с 01.07.2026"),
                text_doc([text]), load_settings())
    assert r.status == "rejected" and r.reason.startswith("no_target")


def test_vat_is_kept_only_where_the_document_states_it():
    s = load_settings()

    def vat(page_text, quote, label="Ставка вознаграждения", said="excluded"):
        r = gate_rate(RateOut(label=label, amount="500", vat=said, page=1, quote=quote), text_doc([page_text]), s)
        assert r.status == "accepted"
        return r.vat

    assert vat("Ставка вознаграждения составляет 500 рублей без НДС.",
               "Ставка вознаграждения составляет 500 рублей без НДС") == "excluded"
    assert vat("Ставка вознаграждения составляет 500 рублей.", "Ставка вознаграждения составляет 500 рублей") == "unknown"
    assert vat("Ставки, руб. без учета НДС\nТариф А 500\nТариф Б 800", "Тариф А 500", label="Тариф А") == "excluded"
    assert vat("Ставка вознаграждения составляет 500 рублей, включая НДС.",
               "Ставка вознаграждения составляет 500 рублей, включая НДС") == "included"
    header = "Период действия ставки | Ставка за одно заявление, руб. без НДС\nс 01.07.2026 по 31.12.2026 | 500"
    assert vat(header, "с 01.07.2026 по 31.12.2026 | 500", label="Ставка за одно заявление", said="unknown") == "excluded"


def test_penalty_is_not_a_rate():
    quote = "За каждый случай неоказания услуг Исполнитель уплачивает Заказчику штраф 5 000 (пять тысяч) рублей"
    r = gate_rate(RateOut(label="штраф за неоказание услуг", amount="5000", page=1, quote=quote),
                  text_doc([quote + "."]), load_settings())
    assert r.status == "rejected" and r.reason.startswith("not_a_rate")


def test_agreement_on_mutual_termination_of_obligations_is_operative():
    text = "1. Стороны договорились о взаимном прекращении обязательств по продвижению услуг с 01.07.2026 г."
    quote = "Стороны договорились о взаимном прекращении обязательств по продвижению услуг с 01.07.2026 г"
    assert gate_op(OpOut(op="suspend_part", effective_from="2026-07-01", page=1, quote=quote), text_doc([text]),
                   load_settings()).status == "accepted"


@pytest.mark.parametrize(("op", "text", "accepted"), [
    ("replace_clause", "1. Изложить пункт 4.2 Договора в следующей редакции: «4.2. Ставка 500 рублей».", True),
    ("replace_clause", "1. Пункт 4.2 Договора читать в новой редакции.", True),
    ("replace_clause", "4. Вознаграждение Дилера определяется по пункту 4.2 Договора.", False),
    ("add_clause", "1. Дополнить Договор пунктом 4.2.1 следующего содержания.", True),
    ("delete_clause", "1. Пункт 7.3 Договора исключить.", True),
    ("delete_clause", "1. Пункт 7.3 Договора признать утратившим силу.", True),
    ("replace_appendix", "1. Приложение № 3 к Договору изложить в редакции Приложения № 1 к настоящему Соглашению.", True),
])
def test_clause_operation_needs_its_own_wording(op, text, accepted):
    quote = text.split(". ", 1)[1].rstrip(".")
    target = {"add_clause": "4.2.1", "delete_clause": "7.3", "replace_appendix": "Приложение 3"}.get(op, "4.2")
    r = gate_op(OpOut(op=op, target=target, page=1, quote=quote), text_doc([text]), load_settings())
    assert (r.status == "accepted") is accepted, r.reason


def test_entry_into_force_is_not_the_date_of_an_operation():
    text = "1. Стороны по взаимному согласию расторгают Договор с 01.09.2026."
    r = apply_gate(Candidate(field="effective_from", value="2026-09-01", kind="date", page=1,
                             quote="Стороны по взаимному согласию расторгают Договор с 01.09.2026"),
                   text_doc([text]), load_settings())
    assert r.status == "rejected" and r.reason.startswith("date_role")


def test_operation_takes_the_start_date_its_own_sentence_states():
    text = "2. С 01.03.2026 прекращаются обязательства Дилера по пункту 1.1.1 Договора.\n3. Соглашение вступает в силу с 16.02.2026."
    r = gate_op(OpOut(op="suspend_part", target="1.1.1", page=1,
                      quote="С 01.03.2026 прекращаются обязательства Дилера по пункту 1.1.1 Договора"),
                text_doc([text]), load_settings())
    assert r.status == "accepted" and str(r.effective_from) == "2026-03-01"


def test_parent_is_the_contract_a_document_belongs_to_not_one_it_mentions():
    s = load_settings()

    def parent(text, quote):
        return apply_gate(Candidate(field="parent_number", value="D990000001-01", kind="doc_number", page=1, quote=quote),
                          text_doc([text]), s)

    mention = ("1. Термины. ДКП – Договор коммерческого представительства, заключенный между Компанией и Партнером "
               "№ D990000001-01 от 15.12.2015.")
    r = parent(mention, "заключенный между Компанией и Партнером № D990000001-01")
    assert r.status == "rejected" and r.reason.startswith("parent_role")
    own = "Дополнительное соглашение № D990000002-01 к Договору № D990000001-01 от 15.12.2015"
    assert parent(own, "к Договору № D990000001-01 от 15.12.2015").status == "accepted"
    change = "Стороны вносят следующие изменения в Договор № D990000001-01 от 15.12.2015:"
    assert parent(change, "изменения в Договор № D990000001-01").status == "accepted"


def test_percent_rate_keeps_its_unit():
    quote = "вознаграждение составляет 60% от оплаченных Клиентом начислений"
    r = gate_rate(RateOut(label="Вознаграждение", amount="60%", page=1, quote=quote), text_doc([quote + "."]),
                  load_settings())
    assert r.status == "accepted" and r.unit == "%"
    rub = gate_rate(RateOut(label="Ставка", amount="500", page=1, quote="Ставка 500 рублей"),
                    text_doc(["Ставка 500 рублей."]), load_settings())
    assert rub.unit == "RUB"


@pytest.mark.parametrize("adjective", ["Агентскому", "Дилерскому", "Лицензионному", "Партнёрскому"])
def test_parent_contract_named_with_an_adjective_is_the_parent(adjective):
    reference = f"к {adjective} договору № D990000001-01 от 10.01.2024 г."
    doc = text_doc([f"Дополнительное соглашение № D990000002-01\n{reference}\nг. Тверь 15.10.2026 г."])
    r = apply_gate(Candidate(field="parent_number", value="D990000001-01", kind="doc_number", page=1, quote=reference),
                   doc, load_settings())
    assert r.status == "accepted", r.reason


@pytest.mark.parametrize("adjective", ["Агентскому", "Дилерскому", "Лицензионному", "Партнёрскому"])
def test_date_of_a_parent_named_with_an_adjective_is_not_the_signing_date(adjective):
    title = f"Дополнительное соглашение № D990000002-01 к {adjective} договору № D990000001-01 от 10.01.2024 г."
    r = apply_gate(Candidate(field="doc_date", value="2024-01-10", kind="date", page=1, quote=title),
                   text_doc([title + "\nг. Тверь 15.10.2026 г."]), load_settings())
    assert r.status == "rejected" and r.reason.startswith("date_role")


def test_document_date_after_a_reference_to_the_document_itself_is_accepted():
    quote = "Приложение № 1 к настоящему Соглашению № D990000002-01 от 15.10.2026"
    r = apply_gate(Candidate(field="doc_date", value="2026-10-15", kind="date", page=1, quote=quote),
                   text_doc([quote + "\nСтавки вознаграждения"]), load_settings())
    assert r.status == "accepted", r.reason


@pytest.mark.parametrize(("quote", "accepted"), [
    ("Настоящее Дополнительное соглашение действует с 01.07.2026", True),
    ("Условия настоящего Дополнительного соглашения применяются с 01.07.2026", True),
    ("Условия, изложенные в настоящем Соглашении, применяются с 01.07.2026", True),
    ("Дополнить Договор пунктом 5.4 с 01.07.2026", False),
    ("Пункт 3.1 Договора изложить в следующей редакции с 01.07.2026", False)])
def test_entry_into_force_is_told_from_an_operation_by_its_verb(quote, accepted):
    r = apply_gate(Candidate(field="effective_from", value="2026-07-01", kind="date", page=1, quote=quote),
                   text_doc([f"5. {quote}."]), load_settings())
    assert (r.status == "accepted") is accepted, r.reason
    assert accepted or r.reason.startswith("date_role")


@pytest.mark.parametrize(("text", "accepted"), [
    ("1. Дополнить Договор пунктом 5.4 следующего содержания.", True),
    ("1. Включить в Договор пункт 5.4 следующего содержания.", True),
    ("1. Пункт 5.4 Договора действует по 31.12.2026 включительно.", False),
    ("1. Пункт 5.4 Договора применяется в редакции Дополнительного соглашения № 1.", False)])
def test_add_clause_needs_a_verb_that_adds(text, accepted):
    quote = text.split(". ", 1)[1].rstrip(".")
    r = gate_op(OpOut(op="add_clause", target="5.4", page=1, quote=quote), text_doc([text]), load_settings())
    assert (r.status == "accepted") is accepted, r.reason
    assert accepted or r.reason.startswith("not_operative")


_REWRITE_3_1 = "1. Пункт 3.1 Договора изложить в следующей редакции: «3.1. Ставка составляет 700 рублей»."
_REWRITE_1_6 = "1. Пункт 1.6 Приложения № 1 к Договору изложить в следующей редакции."


@pytest.mark.parametrize(("op", "text", "quote", "target", "accepted"), [
    ("replace_clause", _REWRITE_3_1, "Пункт 3.1 Договора изложить в следующей редакции", "9.4", False),
    ("replace_clause", _REWRITE_3_1, "Пункт 3.1 Договора изложить в следующей редакции", "3.1", True),
    ("replace_clause", _REWRITE_1_6, "Пункт 1.6 Приложения № 1 к Договору изложить в следующей редакции",
     "Приложение 2/1.6", False),
    ("replace_clause", _REWRITE_1_6, "Пункт 1.6 Приложения № 1 к Договору изложить в следующей редакции",
     "Приложение 1/1.6", True),
    # Only the clause the sentence stands in names the target: a clause heading in that appendix.
    ("replace_clause", "Приложение № 1\n1.6. Вознаграждение Агента пересматривается. Изложить пункт в следующей редакции.",
     "Изложить пункт в следующей редакции", "Приложение 1/1.6", True),
    ("replace_appendix", "1. Приложение № 7 к Договору изложить в редакции Приложения № 1 к настоящему Соглашению.",
     "Приложение № 7 к Договору изложить в редакции Приложения № 1 к настоящему Соглашению", "Приложение 7", True)])
def test_clause_operation_target_must_be_named_by_its_sentence_or_clause(op, text, quote, target, accepted):
    r = gate_op(OpOut(op=op, target=target, page=1, quote=quote), text_doc([text]), load_settings())
    assert (r.status == "accepted") is accepted, r.reason
    assert accepted or r.reason.startswith("target_not_named")


_REWRITE_7_2 = "Пункт 7.2 Договора изложить в следующей редакции"


@pytest.mark.parametrize(("operation", "wording", "accepted"), [
    (_REWRITE_7_2, "Арендатор вправе расторгнуть Договор, если Арендодатель нарушит срок передачи", True),
    (_REWRITE_7_2, "В случае просрочки оплаты Арендатор уплачивает пени", True),
    (_REWRITE_7_2, "Арендатор вправе отказаться от Договора. Уведомление направляется за 30 дней", True),
    ("Стороны вправе изложить пункт 7.2 Договора в следующей редакции", "Арендная плата вносится ежемесячно", False)])
def test_permissive_words_count_only_outside_the_quoted_new_wording(operation, wording, accepted):
    text = f"1. {operation}: «7.2. {wording}»."
    r = gate_op(OpOut(op="replace_clause", target="7.2", page=1, quote=operation), text_doc([text]), load_settings())
    assert (r.status == "accepted") is accepted, r.reason
    assert accepted or r.reason.startswith("not_operative")


@pytest.mark.parametrize(("sentence", "vat"), [
    ("Арендная плата составляет 120 000 рублей в месяц, не включая НДС", "excluded"),
    ("Арендная плата составляет 120 000 рублей в месяц, НДС не включён", "excluded"),
    ("Арендная плата составляет 120 000 рублей в месяц, включая НДС", "included"),
    ("Арендная плата составляет 120 000 рублей в месяц по фиксированной цене включая НДС", "included")])
def test_vat_not_included_is_read_as_excluded(sentence, vat):
    r = gate_rate(RateOut(label="Арендная плата", amount="120 000", vat="excluded", page=1, quote=sentence),
                  text_doc([f"3.1. {sentence}."]), load_settings())
    assert (r.status, r.vat) == ("accepted", vat)


@pytest.mark.parametrize(("sentence", "amount", "unit"), [
    ("Вознаграждение Агента составляет 60 (шестьдесят) процентов от суммы платежей", "60", "%"),
    ("Вознаграждение Агента составляет 60 (шестьдесят) % от суммы платежей", "60", "%"),
    ("Вознаграждение Агента составляет 1 500 (одна тысяча пятьсот) рублей за абонента", "1 500", "RUB")])
def test_percent_written_with_the_amount_in_words_keeps_its_unit(sentence, amount, unit):
    r = gate_rate(RateOut(label="Вознаграждение Агента", amount=amount, page=1, quote=sentence),
                  text_doc([f"4.1. {sentence}."]), load_settings())
    assert (r.status, r.unit) == ("accepted", unit)


def test_an_invalidation_keeps_no_date_because_the_fold_never_dates_one():
    """«Признать ДС № … от «02» февраля 2026 г. недействительным»: the date is the cancelled document's own; a date the
    model attaches to an invalidation is not used by the fold and must not cost the operation itself."""
    text = ("1. Признать Дополнительное соглашение № D990000003-01 от «02» февраля 2026 г. о расторжении Договора "
            "недействительным.")
    op = OpOut(op="invalidate_document", target="D990000003-01", effective_from="2026-02-02", page=1,
               quote="Признать Дополнительное соглашение № D990000003-01 от «02» февраля 2026 г. о расторжении Договора "
                     "недействительным")
    r = gate_op(op, text_doc([text]), load_settings())
    assert (r.status, r.effective_from, r.effective_to) == ("accepted", None, None)


# --- wrong values the code can refuse or correct (fix/wrong-values-p4) ---------------------------------------------

_TABLE = text_doc([("Таблица № 2\nНазвание Группы тарифных планов % от суммы оплаченных Клиентом начислений "
                    "Ставка вознаграждения (руб. без НДС)\nТариф Старт 45% 2000\nТариф Плюс 50% 2400")])


def test_a_percentage_has_no_vat_even_under_a_vat_header():
    s = load_settings()
    percent = gate_rate(RateOut(label="Тариф Старт, %", amount="45%", vat="excluded", page=1, quote="Тариф Старт 45% 2000"), _TABLE, s)
    rubles = gate_rate(RateOut(label="Тариф Старт, ставка", amount="2000", vat="excluded", page=1, quote="Тариф Старт 45% 2000"),
                       _TABLE, s)
    assert (percent.status, percent.unit, percent.vat) == ("accepted", "%", "unknown")
    assert (rubles.status, rubles.unit, rubles.vat) == ("accepted", "RUB", "excluded")



def test_a_zero_amount_is_refused_not_corrected():
    """«60%» misread as «б0%» on a scan: the code does not turn the letter back into a digit."""
    scan = text_doc(["- в размере б0%% от оплаченных начислений по пакетному предложению за каждого Клиента"])
    row = gate_rate(RateOut(label="Услуги Агента", amount="0", page=1,
                            quote="в размере б0%% от оплаченных Клиентом начислений"), scan, load_settings())
    assert row.status != "accepted" and row.reason.split(":")[0] == "zero_amount"


@pytest.mark.parametrize("quote", ["в лице ______, действующего на основании доверенности № ___ от 15.01.2021",
                                   "в лице Иванова И. И., действующего на основании приказа № 12 от 15.01.2021"])
def test_the_date_of_a_representatives_authority_is_not_the_document_date(quote):
    from contract_analyst.gate.checks import check_date_role

    got = check_date_role("doc_date", "2021-01-15", quote)
    assert (got.ok, got.code) == (False, "date_role")


@pytest.mark.parametrize("quote", ["г. Москва 15.01.2021", "Дополнительное соглашение № D990000002-01 от 15.01.2021"])
def test_a_document_date_is_still_accepted(quote):
    from contract_analyst.gate.checks import check_date_role

    assert check_date_role("doc_date", "2021-01-15", quote).ok


def test_a_date_right_after_a_masked_representative_is_refused_from_the_page_text():
    """The words «действующего на основании доверенности № … от» are masked together with the name, and the model's
    quote starts at the date: the preamble before the quote shows whose date it is."""
    page = text_doc([("Дополнительное соглашение № D990000002-01 к Договору № D990000001-01 от 19.08.2020 г.\n"
                      "АО «Альфа-Связь», в лице ведущего специалиста отдела продаж - - - - 02.06.2023г., "
                      "с одной стороны, и ООО «Ромашка», с другой стороны, заключили настоящее соглашение.")])
    got = apply_gate(Candidate(field="doc_date", value="2023-06-02", kind="date", page=1,
                               quote="02.06.2023г., с одной стороны, и"), page, load_settings())
    assert (got.status, got.reason.split(":")[0]) == ("rejected", "date_role")


def test_a_rate_whose_percent_sign_stands_only_in_the_column_header_is_a_percentage():
    """«% от суммы оплаченных Клиентом начислений» heads the column; the row reads «Тариф Старт / 45 / 2000» and the model's label
    says it is a percentage."""
    table = text_doc([("Таблица № 2\nНазвание Группы тарифных планов\n% от суммы оплаченных Клиентом начислений\n"
                       "Ставка вознаграждения (руб. без НДС)\nТариф Старт\n45\n2000")])
    row = gate_rate(RateOut(label="Тариф Старт — процент от суммы оплаченных Клиентом начислений", amount="45",
                            vat="excluded", page=1, quote="Тариф Старт\n45\n2000"), table, load_settings())
    assert (row.status, row.unit, row.vat) == ("accepted", "%", "unknown")


_TARGETS = ("2.2. Компания устанавливает Перечень Плановых показателей к выполнению.\nТаблица № 5\n"
            "Плановый показатель | Целевое значение выполнения планового показателя\n"
            "План подключений 95%\nПлан подключений с переносом номера 100%\nТребования к ассортименту 95%\n"
            "2.3. Компания выплачивает Агенту вознаграждение при выполнении Плановых показателей:\n"
            "% от оплаченных Клиентом начислений (при выполнении плановых показателей №1): 45\n"
            "Ставка вознаграждения (руб. без НДС)\nТариф Старт 300\n"
            "Увеличение значения X за каждый выполненный плановый показатель: 2 п.п.")


@pytest.mark.parametrize(("label", "amount", "quote"), [
    ("План подключений", "95%", "План подключений 95%"),
    ("План подключений с переносом номера", "100%", "План подключений с переносом номера 100%"),
    ("Требования к ассортименту", "95%", "Требования к ассортименту 95%")])
def test_a_sales_target_is_not_a_rate(label, amount, quote):
    """A target («целевое значение выполнения планового показателя») is the share of the plan to reach, not a price."""
    row = gate_rate(RateOut(label=label, amount=amount, page=1, quote=quote), text_doc([_TARGETS]), load_settings())
    assert (row.status, row.reason.split(":")[0]) == ("rejected", "not_a_rate")


@pytest.mark.parametrize(("label", "amount", "quote", "unit"), [
    ("% от оплаченных Клиентом начислений (при выполнении плановых показателей №1)", "45",
     "% от оплаченных Клиентом начислений (при выполнении плановых показателей №1): 45", "%"),
    ("Тариф Старт", "300", "Тариф Старт 300", "RUB"),
    ("Тариф Старт (при выполнении плановых показателей №1)", "300", "Тариф Старт 300", "RUB"),
    ("Увеличение значения X за каждый выполненный плановый показатель, п.п.", "2",
     "Увеличение значения X за каждый выполненный плановый показатель: 2 п.п.", "RUB")])
def test_a_reward_paid_when_targets_are_met_is_still_a_rate(label, amount, quote, unit):
    row = gate_rate(RateOut(label=label, amount=amount, page=1, quote=quote), text_doc([_TARGETS]), load_settings())
    assert (row.status, row.unit) == ("accepted", unit), row.reason


@pytest.mark.parametrize("op", ["other", "replace_clause"])
def test_the_entry_into_force_of_a_supplement_is_not_an_operation(op):
    quote = "Настоящее Соглашение вступает в силу с 01.04.2024"
    r = gate_op(OpOut(op=op, target="5", page=1, quote=quote), text_doc([f"5. {quote}."]), load_settings())
    assert (r.status, r.reason.split(":")[0]) == ("rejected", "not_an_operation")


def test_an_operation_that_also_states_its_entry_into_force_is_kept():
    quote = "Пункт 3.1 Договора изложить в следующей редакции, Соглашение вступает в силу с 01.04.2024"
    r = gate_op(OpOut(op="replace_clause", target="3.1", page=1, quote=quote), text_doc([f"1. {quote}."]),
                load_settings())
    assert r.status == "accepted", r.reason


@pytest.mark.parametrize(("amount", "quote", "rubles"), [
    ("10", "вознаграждение составляет 10 (Десять) копеек (без учета НДС)", "0.10"),
    ("10", "но не менее 10 (Десяти) копеек (без учета НДС)", "0.10"),
    ("10", "вознаграждение составляет 10 рублей 50 копеек", "10.50"),
    ("6500", "Стоимость доставки составляет 6 500 (шесть тысяч пятьсот) рублей 00 копеек", "6500"),
    ("400,00", "вознаграждение в размере 400,00 (четыреста рублей 00 коп.) рублей", "400")])
def test_an_amount_in_kopecks_is_written_in_rubles(amount, quote, rubles):
    row = gate_rate(RateOut(label="Вознаграждение", amount=amount, page=1, quote=quote), text_doc([quote + "."]),
                    load_settings())
    assert (row.status, row.unit, row.amount) == ("accepted", "RUB", Decimal(rubles)), row.reason


def test_the_first_part_of_a_reward_is_how_it_is_paid_not_a_rate():
    quote = ("в месяце, следующем за месяцем Регистрации, вознаграждение выплачивается в размере 10 (Десяти) копеек "
             "(Первая часть вознаграждения)")
    row = gate_rate(RateOut(label="Первая часть вознаграждения", amount="10", page=1, quote=quote),
                    text_doc([quote + "."]), load_settings())
    assert (row.status, row.reason.split(":")[0]) == ("rejected", "not_a_rate")


def _row(text: str, label: str, amount: str, quote: str):
    return gate_rate(RateOut(label=label, amount=amount, page=1, quote=quote), text_doc([text]), load_settings())


def test_vat_in_a_table_header_far_above_the_row_applies_to_it():
    filler = "\n".join(f"Тарифный план группы {i}\n{100 + i}" for i in range(60))
    text = f"1. Изложить Таблицу 1 в следующей редакции:\nТаблица №1\nНазвание\nСтавка, руб. без НДС\n{filler}\nИнтернет\n500"
    assert text.find("Интернет") - text.find("без НДС") > 1000
    assert _row(text, "Интернет", "500", "Интернет 500").vat == "excluded"


def test_vat_is_found_when_the_cells_of_a_row_stand_on_separate_lines():
    text = "Таблица №1\nСтавка вознаграждения (руб.\nбез НДС)\nТариф Старт\n270\n500\nТариф Макс\n675\n1200"
    assert _row(text, "Тариф Макс", "1200", "Тариф Макс 675 1200").vat == "excluded"


def test_vat_of_one_table_does_not_pass_to_the_next():
    text = ("Таблица №1\nСтавка, руб. с НДС\nТариф Старт 300\nТаблица №2\nДоплата за подключение\nТариф Старт 150")
    assert _row(text, "Тариф Старт", "300", "Тариф Старт 300").vat == "included"
    assert _row(text, "Доплата за подключение, Тариф Старт", "150", "Тариф Старт 150").vat == "unknown"


@pytest.mark.parametrize(("amount", "quote"), [
    ("0,5", "Исполнитель уплачивает штраф в размере 0,5% от суммы вознаграждения за отчётный месяц"),
    ("3000", "но не менее 3 000 (трёх тысяч) рублей за каждый случай нарушения"),
    ("2000", "Агент возмещает 2 000 рублей за каждое нарушение порядка оформления договоров")])
def test_a_penalty_counted_from_the_reward_is_not_a_rate(amount, quote):
    row = _row(quote + ".", "Вознаграждение Агента", amount, quote)
    assert (row.status, row.reason.split(":")[0]) == ("rejected", "not_a_rate")


def test_a_reward_that_mentions_a_penalty_elsewhere_is_still_a_rate():
    quote = "Вознаграждение составляет 300 рублей за каждого Абонента, штраф начисляется отдельно"
    assert _row(quote + ".", "Вознаграждение", "300", quote).status == "accepted"


@pytest.mark.parametrize(("text", "quote", "amount", "vat"), [
    ("3.1. Стоимость услуги составляет 3 100 (три тысячи сто) рублей. Указанные суммы включают НДС.",
     "3 100 (три тысячи сто) рублей", "3100", "included"),
    ("2.2. Стоимость доставки составляет 2 400 рублей 00 копеек, кроме того НДС 20%.",
     "2 400 рублей 00 копеек", "2400", "excluded"),
    ("4.1. Цена работ — 15 000 рублей, сверх того НДС по ставке 20%.", "15 000 рублей", "15000", "excluded"),
    ("3.1. Стоимость Услуг составляет 70 000 рублей в месяц. НДС не облагается.",
     "Стоимость Услуг составляет 70 000 рублей в месяц", "70000", "excluded"),
    ("3.1. Стоимость Услуг составляет 70 000 рублей в месяц. Суммы не включают НДС.",
     "Стоимость Услуг составляет 70 000 рублей в месяц", "70000", "excluded")])
def test_vat_stated_after_the_amount_in_its_clause_applies_to_it(text, quote, amount, vat):
    assert _row(text, "Стоимость", amount, quote).vat == vat


def test_vat_stated_in_the_next_clause_does_not_apply():
    text = "3.1. Стоимость уборки 1 000 рублей.\n3.2. Стоимость мойки окон 2 000 рублей, кроме того НДС."
    assert _row(text, "Стоимость уборки", "1000", "Стоимость уборки 1 000 рублей").vat == "unknown"


@pytest.mark.parametrize("quote", [
    "Пункт 2.3 Договора в редакции настоящего Соглашения применяется с 01 июня 2026 года",
    "Прайс-лист в новой редакции применяется к заказам, оформленным начиная с 01.02.2027"])
def test_the_date_a_change_applies_from_is_not_an_operation_of_its_own(quote):
    r = gate_op(OpOut(op="other", target="2.3", page=1, quote=quote), text_doc([f"3. {quote}."]), load_settings())
    assert (r.status, r.reason.split(":")[0]) == ("rejected", "not_an_operation")


def test_a_clause_that_stops_applying_is_still_an_operation():
    quote = "Пункт 2.3 Договора не применяется с 01 июня 2026 года"
    r = gate_op(OpOut(op="other", target="2.3", page=1, quote=quote), text_doc([f"3. {quote}."]), load_settings())
    assert r.reason is None or not r.reason.startswith("not_an_operation"), r.reason


@pytest.mark.parametrize(("text", "vat"), [
    ("3.1. Цена работ — 15 000 рублей, в том числе налог на добавленную стоимость 20%.", "included"),
    ("3.1. Цена работ — 15 000 рублей без учёта налога на добавленную стоимость.", "excluded"),
    ("3.1. Цена работ — 15 000 рублей. Налог на добавленную стоимость не облагается.", "excluded")])
def test_vat_written_out_in_full_is_read_like_its_abbreviation(text, vat):
    assert _row(text, "Цена работ", "15000", "Цена работ — 15 000 рублей").vat == vat

