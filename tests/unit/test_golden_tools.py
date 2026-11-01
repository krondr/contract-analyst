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


from contract_analyst.eval.golden import GoldenDocument, GoldenTrap, GoldenValue
from contract_analyst.eval.golden_tools import build_review_page, check_golden, diff_golden, fold_mismatches

PAGE = "Договор № D990000001-01\nг. Тверь 19.08.2020 г.\nСтавка составляет 500 рублей без НДС."


def _gdoc(**kw):
    base = {"file": "m.pdf", "doc_id": "d1", "kind": "main",
                "number": GoldenValue(value="D990000001-01", page=1, quote="Договор № D990000001-01"),
                "doc_date": GoldenValue(value="2020-08-19", page=1, quote="г. Тверь 19.08.2020 г.")}
    base.update(kw)
    return GoldenDocument(**base)


def _gs(**kw):
    chain = GoldenChain(contract_number="D990000001-01", main_doc_id="d1", order=["d1"],
                        status_at={date(2026, 10, 30): "active"},
                        rates_at={date(2026, 10, 30): [GoldenRate(label="Ставка", amount=Decimal(500))]})
    rates = [GoldenRate(label="Ставка", amount=Decimal(500), page=1, quote="Ставка составляет 500 рублей без НДС")]
    base = {"as_of_dates": [date(2026, 10, 30)], "documents": [_gdoc(rates=rates)], "chains": [chain]}
    base.update(kw)
    return GoldenSet(**base)


def test_diff_finds_disagreement():
    other = _gs(documents=[_gdoc(number=GoldenValue(value="D990000009-01", page=1, quote="x"))])
    assert "documents[d1].number" in [d.path for d in diff_golden(_gs(), other)]


def test_diff_tolerates_paraphrased_terms():
    a = _gs(documents=[_gdoc(terms={"validity": GoldenValue(value="до 31.12.2026 с пролонгацией")})])
    b = _gs(documents=[_gdoc(terms={"validity": GoldenValue(value="действует до 31.12.2026 с автоматической пролонгацией")})])
    assert not [d for d in diff_golden(a, b) if "terms" in d.path]


def test_check_passes_consistent_set():
    assert check_golden(_gs(), {"d1": text_doc([PAGE], doc_id="d1")}) == []


def test_check_catches_missing_quote():
    bad = _gs(documents=[_gdoc(city=GoldenValue(value="Псков", page=1, quote="г. Псков"))])
    errors = check_golden(bad, {"d1": text_doc([PAGE], doc_id="d1")})
    assert any("quote" in e and "city" in e for e in errors)


def test_fold_mismatch_needs_adjudication_and_is_not_an_error():
    wrong = _gs()
    wrong.chains[0].rates_at[date(2026, 10, 30)] = [GoldenRate(label="Ставка", amount=Decimal(900))]
    assert check_golden(wrong, {"d1": text_doc([PAGE], doc_id="d1")}) == []
    (item,) = fold_mismatches(wrong)
    assert item.startswith("chains[D990000001-01].rates_at[2026-10-30]") and "900" in item and "500" in item
    assert fold_mismatches(_gs()) == []


def test_check_catches_chain_number_no_document_has():
    typo = _gs(chains=[_gs().chains[0].model_copy(update={"contract_number": "D990000777-01"})])
    errors = check_golden(typo, {"d1": text_doc([PAGE], doc_id="d1")})
    assert errors == ["chains[D990000777-01]: no document in order has this number or refers to it"]


def _cli_check(tmp_path, monkeypatch, gs):
    from typer.testing import CliRunner

    from contract_analyst import cli

    monkeypatch.setattr(cli, "_read_all", lambda data: [text_doc([PAGE], doc_id="d1")])
    dump_golden(gs, tmp_path / "g.yaml")
    return CliRunner().invoke(cli.app, ["golden", "check", str(tmp_path / "g.yaml"), "--data", str(tmp_path)])


def test_cli_golden_check_fails_only_on_errors(tmp_path, monkeypatch):
    wrong = _gs()
    wrong.chains[0].rates_at[date(2026, 10, 30)] = [GoldenRate(label="Ставка", amount=Decimal(900))]
    res = _cli_check(tmp_path, monkeypatch, wrong)
    assert res.exit_code == 0 and "needs adjudication: chains[D990000001-01].rates_at[2026-10-30]" in res.output
    bad = _gs(documents=[_gdoc(city=GoldenValue(value="Псков", page=1, quote="г. Псков"))])
    assert _cli_check(tmp_path, monkeypatch, bad).exit_code == 1


def test_check_requires_traps():
    errors = check_golden(_gs(traps=[GoldenTrap(trap_id="T2", kind="doc_kind_is")]), {"d1": text_doc([PAGE], doc_id="d1")},
                          required_traps={"T1", "T2"})
    assert "missing trap T1" in errors


def test_review_page_has_all_disputed_and_sample(tmp_path):
    from contract_analyst.eval.golden_tools import GoldenDiff

    terms = {f"t{i}": GoldenValue(value=f"условие {i}", page=1, quote=f"цитата {i}") for i in range(10)}
    gs = _gs(documents=[_gdoc(terms=terms)])
    disputed = [GoldenDiff(path="documents[d1].city", a="Тверь", b="Псков")]
    html = build_review_page(gs, disputed, tmp_path / "r.html", sample_share=0.2).read_text()
    assert "documents[d1].city" in html and "Псков" in html
    assert html.count('class="sample"') >= 2


def test_cli_golden_diff_writes_json(tmp_path):
    import json

    from typer.testing import CliRunner

    from contract_analyst.cli import app

    dump_golden(_gs(), tmp_path / "a.yaml")
    dump_golden(_gs(documents=[_gdoc(city=GoldenValue(value="Тверь", page=1, quote="г. Тверь"))]), tmp_path / "b.yaml")
    res = CliRunner().invoke(app, ["golden", "diff", str(tmp_path / "a.yaml"), str(tmp_path / "b.yaml"),
                                   "--out", str(tmp_path / "d.json")])
    assert res.exit_code == 0 and json.loads((tmp_path / "d.json").read_text())[0]["path"] == "documents[d1].city"
