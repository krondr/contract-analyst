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
from tests.fixtures.builders import text_doc

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
