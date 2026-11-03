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


from contract_analyst.eval.golden import GoldenDocument, GoldenOp, GoldenTrap, GoldenValue
from contract_analyst.eval.golden_tools import (
    build_review_page,
    check_golden,
    diff_golden,
    fold_mismatches,
    golden_fold,
)

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


def test_diff_flags_terms_that_differ_in_a_date_or_a_key_word():
    def gs(validity, payment):
        return _gs(documents=[_gdoc(terms={"validity": GoldenValue(value=validity),
                                           "payment_terms": GoldenValue(value=payment)})])

    a = gs("действует до 31 декабря 2026 года", "оплата ежемесячно до 10 числа")
    b = gs("действует до 31 декабря 2025 года", "оплата ежеквартально до 10 числа")
    assert {d.path for d in diff_golden(a, b)} == {"documents[d1].terms.validity", "documents[d1].terms.payment_terms"}
    assert not diff_golden(a, gs("действует до 31.12.2026", "ежемесячная оплата до 10 числа"))


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


def test_golden_values_carry_the_clause_they_quote():
    from contract_analyst.eval.golden_tools import _to_record

    page = "Договор № D990000001-01\n12.1. Уведомления направляются по факсу.\n12.2. Копии имеют силу оригинала."
    gdoc = _gdoc(terms={"notification_channel": GoldenValue(value="по факсу", page=1,
                                                            quote="Уведомления направляются по факсу.")})
    record = _to_record(gdoc, text_doc([page], doc_id="d1"))
    assert record.terms["notification_channel"].source.clause_id == "12.1"


def test_diff_flags_vat_and_ignores_how_a_clause_is_spelled():
    from contract_analyst.eval.golden import GoldenOp

    def gs(vat, target):
        rates = [GoldenRate(label="Ставка", amount=Decimal(500), vat=vat, page=1, quote="Ставка 500 рублей")]
        return _gs(documents=[_gdoc(rates=rates, ops=[GoldenOp(op="replace_clause", target=target)])])

    assert [d.path for d in diff_golden(gs("excluded", "п. 3.1"), gs("included", "пункт 3.1"))] == ["documents[d1].rates"]


def test_fold_check_reads_clauses_from_the_documents():
    from contract_analyst.eval.golden import GoldenOp

    main_page = ("Договор № D990000001-01\nг. Тверь 19.08.2020 г.\n12.1. Уведомления направляются по факсу в течение 3 дней.\n"
                 "12.2. Копии по электронной почте имеют силу оригинала.")
    sup_page = ("Дополнительное соглашение № D990000002-01 к Договору № D990000001-01 от 19.08.2020\n"
                "1. Изложить пункт 12.2 Договора в редакции: копии через ЭДО в течение 1 дня имеют силу оригинала.")
    main = _gdoc(terms={"notification_channel": GoldenValue(value="по факсу в течение 3 дней", page=1,
                                                            quote="Уведомления направляются по факсу в течение 3 дней.")})
    sup = _gdoc(file="s.pdf", doc_id="d2", kind="supplement",
                number=GoldenValue(value="D990000002-01", page=1, quote="Дополнительное соглашение № D990000002-01"),
                doc_date=GoldenValue(status="absent"),
                parent_number=GoldenValue(value="D990000001-01", page=1, quote="к Договору № D990000001-01"),
                effective_from=GoldenValue(value="2024-01-01", page=1, quote="Дополнительное соглашение"),
                terms={"notification_channel": GoldenValue(value="копии через ЭДО в течение 1 дня", page=1,
                                                           quote="копии через ЭДО в течение 1 дня")},
                ops=[GoldenOp(op="replace_clause", target="12.2")])
    chain = GoldenChain(contract_number="D990000001-01", main_doc_id="d1", order=["d1", "d2"],
                        status_at={date(2026, 10, 30): "active"},
                        terms_at={date(2026, 10, 30): {"notification_channel": "по факсу в течение 3 дней; копии через ЭДО в течение 1 дня"}})
    gs = _gs(documents=[main, sup], chains=[chain])
    docs = {"d1": text_doc([main_page], doc_id="d1"), "d2": text_doc([sup_page], doc_id="d2")}
    assert fold_mismatches(gs, docs) == []
    assert fold_mismatches(gs) != []  # without the texts no clause is known and the supplement replaces


# --- the code fold on the golden set itself, scored like a run ------------------------------------------------------

AS_OF = date(2026, 10, 30)
MAIN_TEXT = ("Договор № D990000001-01\nг. Тверь 19.08.2020 г.\n3.1. Ставка составляет 500 рублей без НДС.\n"
             "12.1. Уведомления направляются по факсу в течение 3 дней.\n"
             "12.2. Копии по электронной почте имеют силу оригинала.")
SUP_TEXT = ("Дополнительное соглашение № D990000002-01 к Договору № D990000001-01 от 19.08.2020\n"
            "1. Изложить пункт 12.2 Договора в редакции: копии через ЭДО в течение 1 дня имеют силу оригинала.\n"
            "2. Изложить пункт 3.1 Договора в редакции: с 01.01.2024 ставка составляет 700 рублей без НДС.")
COMBINED = "по факсу в течение 3 дней; копии через ЭДО в течение 1 дня"


def _amended(label="Ставка"):
    """A contract and a supplement from 01.01.2024 that rewrites clauses 12.2 and 3.1; the new rate is `label`."""
    main = _gdoc(terms={"notification_channel": GoldenValue(value="по факсу в течение 3 дней", page=1,
                                                            quote="Уведомления направляются по факсу в течение 3 дней.")},
                 rates=[GoldenRate(label="Ставка", amount=Decimal(500), vat="excluded", valid_from=date(2020, 8, 19),
                                   page=1, quote="Ставка составляет 500 рублей без НДС.")])
    sup = _gdoc(file="s.pdf", doc_id="d2", kind="supplement",
                number=GoldenValue(value="D990000002-01", page=1, quote="Дополнительное соглашение № D990000002-01"),
                doc_date=GoldenValue(status="absent"),
                parent_number=GoldenValue(value="D990000001-01", page=1, quote="к Договору № D990000001-01"),
                effective_from=GoldenValue(value="2024-01-01", page=1, quote="С 01.01.2024"),
                terms={"notification_channel": GoldenValue(value="копии через ЭДО в течение 1 дня", page=1,
                                                           quote="копии через ЭДО в течение 1 дня")},
                rates=[GoldenRate(label=label, amount=Decimal(700), vat="excluded", valid_from=date(2024, 1, 1),
                                  page=1, quote="с 01.01.2024 ставка составляет 700 рублей без НДС.")],
                ops=[GoldenOp(op="replace_clause", target="12.2"), GoldenOp(op="replace_clause", target="3.1")])
    in_force = [GoldenRate(label=label, amount=Decimal(700), vat="excluded", valid_from=date(2024, 1, 1))]
    chain = GoldenChain(contract_number="D990000001-01", main_doc_id="d1", order=["d1", "d2"],
                        status_at={AS_OF: "active"}, rates_at={AS_OF: in_force},
                        terms_at={AS_OF: {"notification_channel": COMBINED}})
    return _gs(documents=[main, sup], chains=[chain])


def _texts():
    return {"d1": text_doc([MAIN_TEXT], doc_id="d1"), "d2": text_doc([SUP_TEXT], doc_id="d2")}


def test_golden_fold_of_a_correct_golden_set_is_perfect():
    fold = golden_fold(_amended(), _texts())
    assert (fold.acc, fold.misses) == (1.0, [])


def test_golden_fold_without_the_texts_cannot_tell_an_addition_from_a_replacement():
    """The golden set gives a value's page and quote, not its clause: without the text the supplement's value
    replaces the contract's instead of joining it (fold_mismatches sees the same)."""
    fold = golden_fold(_amended())
    assert fold.acc == 2 / 3
    assert [(m["point"], m["reason"], m["expected"], m["got"]) for m in fold.misses] == [
        ("terms.notification_channel", "term_different", "3; 1", "1")]


def test_golden_fold_retires_a_rewritten_clause_s_rate_by_its_source_as_a_run_s_fold_does():
    """A run's rate rows carry the clause they were read from, so clause 3.1 in new wording takes the old rate along
    even when the new one has another label: the golden rows need that source too."""
    fold = golden_fold(_amended(label="Арендная плата"), _texts())
    assert (fold.acc, fold.misses) == (1.0, [])


def test_golden_fold_dates_a_row_without_its_own_period_by_its_document_like_a_run():
    """The scoring gives a folded row without its own period its document's (2020-08-19 here) but leaves a golden
    chain row as written, so this point is lost however well the documents are read; fold_mismatches compares the
    raw rows and does not see it."""
    fold = golden_fold(_gs())
    assert fold.acc == 0.5 and fold_mismatches(_gs()) == []
    assert [(m["point"], m["reason"], m["expected"], m["got"]) for m in fold.misses] == [
        ("rates", "rates_period", "500 unknown …–…", "500 unknown 2020-08-19–…")]


def _run_with_texts(run_dir, documents):
    """A finished run's knowledge base holding only the documents' page texts."""
    from contract_analyst.kb.store import KnowledgeBase

    KnowledgeBase(run_dir / "kb").save(documents=documents, records={}, chains=[], consolidated={}, clusters=[],
                                       decisions=[])
    return run_dir


def test_scores_carry_the_golden_fold_with_the_texts_the_run_stored(tmp_path):
    from contract_analyst.eval.metrics import score_run

    s = score_run(_run_with_texts(tmp_path / "texts", _texts()), _amended())
    assert (s.golden_fold_acc, s.golden_fold_misses) == (1.0, [])
    assert s.consolidation_acc == 0.0  # the run itself folded nothing: the golden fold does not depend on it
    bare = score_run(_run_with_texts(tmp_path / "bare", {}), _amended())  # no page texts: clauses unknown
    assert bare.golden_fold_acc == 2 / 3 and [m["reason"] for m in bare.golden_fold_misses] == ["term_different"]
