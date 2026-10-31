from datetime import date
from decimal import Decimal

from contract_analyst.eval.golden import GoldenChain, GoldenRate, GoldenSet, dump_golden, load_golden
from contract_analyst.eval.golden_tools import prefill
from tests.fixtures.builders import text_doc

SUPPLEMENT = ("Дополнительное соглашение № D990000002-01\n"
              "к Договору № D990000001-01 от «21» января 2021 г.\nг. Тверь «07» июля 2021 г.\n"
              "1. Изложить п. 3.1 Договора в следующей редакции.")


def test_prefill_takes_number_and_date_with_quotes():
    gs = prefill([text_doc([SUPPLEMENT], name="ДС.pdf")])
    (doc,) = gs.documents
    assert doc.kind == "supplement" and doc.file == "ДС.pdf"
    assert doc.number.value == "D990000002-01" and doc.number.page == 1 and "D990000002-01" in doc.number.quote
    assert doc.parent_date.value == "2021-01-21" and "января 2021" in doc.parent_date.quote
    assert doc.doc_date.value == "2021-07-07" and doc.city.value == "Тверь"


def test_yaml_roundtrip_keeps_dates_and_decimals(tmp_path):
    gs = GoldenSet(as_of_dates=[date(2026, 10, 30)], chains=[GoldenChain(
        contract_number="D990000001-01", order=["a"],
        status_at={date(2026, 10, 30): "active"},
        rates_at={date(2026, 10, 30): [GoldenRate(label="Тариф А", amount=Decimal("330.5"))]})])
    dump_golden(gs, tmp_path / "g.yaml")
    back = load_golden(tmp_path / "g.yaml")
    assert back == gs and back.chains[0].rates_at[date(2026, 10, 30)][0].amount == Decimal("330.5")
