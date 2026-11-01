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


def test_rate_quote_reanchored_to_the_line_with_label_and_amount(doc):
    row = gate_rate(RateOut(label="Тариф А", amount="700", page=1, quote="Тариф А"), doc, load_settings())
    assert row.status == "accepted" and "700" in row.source.quote and row.reason == "quote_reanchored"


def test_rate_not_reanchored_when_no_line_has_both(doc):
    row = gate_rate(RateOut(label="Тариф Б", amount="700", page=1, quote="Тариф Б"), doc, load_settings())
    assert row.status == "rejected"


def test_rate_reanchored_in_table_with_cells_on_separate_lines():
    table = text_doc(["Таблица №1\nСтавка (руб. без НДС) с 01.07.2026\nс 01.01.2027\nТариф Старт\n270\n500\nОптима\n330,5\n500\n"])
    row = gate_rate(RateOut(label="Оптима", amount="500", valid_from="2027-01-01", page=1,
                            quote="Ставка (руб. без НДС) с 01.07.2026"), table, load_settings())
    assert row.status == "accepted" and row.source.quote == "Оптима 330,5 500"
    wrong = gate_rate(RateOut(label="Оптима", amount="270", page=1, quote="Ставка"), table, load_settings())
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
    doc = text_doc(["Ставка, руб. без НДС с 01.07.2026 с 01.01.2027\nТариф Старт 270 500\nОптима 330,5 500"])
    s = load_settings()
    r = gate_rate(RateOut(label="Тариф Старт", amount="500", valid_from="2027-01-01", page=1, quote="Тариф Старт 270 500"), doc, s)
    assert r.status == "conflict" and r.reason.startswith("ambiguous_number")
    ok = gate_rate(RateOut(label="Оптима", amount="330,5", valid_from="2026-07-01", page=1,
                           quote="Оптима 330,5 500"), doc, s)
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
