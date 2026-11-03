import re
from datetime import date
from decimal import Decimal

import pytest

from contract_analyst.assemble.chains import build_chains
from contract_analyst.assemble.consolidate import consolidate
from contract_analyst.eval.golden import (
    GoldenChain,
    GoldenCluster,
    GoldenDocument,
    GoldenOp,
    GoldenRate,
    GoldenSet,
    GoldenTrap,
    GoldenValue,
)
from contract_analyst.eval.metrics import (
    normalize_target,
    same_value,
    score_consolidation,
    score_run,
    similar_text,
)
from contract_analyst.eval.traps import check_traps
from contract_analyst.kb.store import KnowledgeBase
from contract_analyst.models import ConsolidatedContract, Extracted, OpType, RateRow, TermState
from tests.fixtures.builders import acc, text_doc
from tests.fixtures.builders import make_record as rec

MAIN, SUP = "D990000001-01", "D990000002-01"
AS_OF = date(2026, 10, 30)


def make_run(tmp_path, main_number=MAIN, tweak=None):
    records = {"m": rec("m", main_number, kind="main", doc_date=date(2021, 1, 21), category="lease"),
               "s": rec("s", SUP, parent=MAIN, effective=date(2026, 7, 1),
                        rates=[("Ставка", 675, date(2026, 7, 1), date(2026, 12, 31)), ("Ставка", 1200, date(2027, 1, 1), None)],
                        ops=[(OpType.SUSPEND_PART, None, date(2026, 8, 1))])}
    if tweak:
        tweak(records)
    chains = build_chains(list(records.values()))
    consolidated = {c.contract_number: consolidate(c, records, AS_OF) for c in chains}
    from contract_analyst.analyze.cluster import cluster_contracts
    from contract_analyst.config import load_settings

    clusters = cluster_contracts(list(consolidated.values()), records, chains, load_settings())
    KnowledgeBase(tmp_path / "kb").save(documents={"m": text_doc(["x"], doc_id="m"), "s": text_doc(["y"], doc_id="s")},
                                        records=records, chains=chains, consolidated=consolidated, clusters=clusters,
                                        decisions=[])
    return tmp_path


def golden():
    docs = [GoldenDocument(file="m.pdf", doc_id="m", kind="main", subject_category="lease",
                           number=GoldenValue(value=MAIN), doc_date=GoldenValue(value="2021-01-21")),
            GoldenDocument(file="s.pdf", doc_id="s", kind="supplement", number=GoldenValue(value=SUP),
                           parent_number=GoldenValue(value=MAIN), effective_from=GoldenValue(value="2026-07-01"),
                           rates=[GoldenRate(label="Ставка", amount=Decimal(675), valid_from=date(2026, 7, 1),
                                             valid_to=date(2026, 12, 31)),
                                  GoldenRate(label="Ставка", amount=Decimal(1200), valid_from=date(2027, 1, 1))],
                           ops=[GoldenOp(op="suspend_part", effective_from=date(2026, 8, 1))])]
    chain = GoldenChain(contract_number=MAIN, main_doc_id="m", order=["m", "s"],
                        status_at={AS_OF: "partially_suspended"},
                        rates_at={AS_OF: [GoldenRate(label="Ставка", amount=Decimal(675), valid_from=date(2026, 7, 1),
                                                     valid_to=date(2026, 12, 31))]})
    traps = [GoldenTrap(trap_id="T5", kind="status_is", contract_number=MAIN,
                        expected={"as_of": "2026-10-30", "status": "partially_suspended"}),
             GoldenTrap(trap_id="T9", kind="rate_planned", contract_number=MAIN,
                        expected={"as_of": "2026-10-30", "label": "Ставка", "amount": "1200"}),
             GoldenTrap(trap_id="T3", kind="order_is", contract_number=MAIN, expected={"order": ["m", "s"]}),
             GoldenTrap(trap_id="T2", kind="doc_kind_is", doc_id="s", expected={"kind": "supplement"}),
             GoldenTrap(trap_id="T10", kind="normalized_number", doc_id="s", expected={"number": SUP}),
             GoldenTrap(trap_id="T6", kind="issue_present", contract_number=MAIN, expected={"code": "missing_main"})]
    return GoldenSet(as_of_dates=[AS_OF], documents=docs, chains=[chain],
                     clusters=[GoldenCluster(category="lease", members=[MAIN])], traps=traps)


def test_perfect_run_scores_full(tmp_path):
    s = score_run(make_run(tmp_path), golden())
    assert s.doc_kind_acc == 1.0 and s.chain_acc == 1.0 and s.critical_accepted_wrong == 0
    assert s.critical_coverage == 1.0 and s.consolidation_acc == 1.0 and s.clusters_match


def test_score_counts_wrong_accepted_critical(tmp_path):
    s = score_run(make_run(tmp_path, main_number="D990000777-01"), golden())
    assert s.critical_accepted_wrong == 1


def test_normalized_comparison():
    assert same_value("ООО  «Ромашка»", "ооо «ромашка»")
    assert same_value(Decimal(700), Decimal("700.0"))
    assert not same_value("D990000001-01", "D990000002-01")


def test_trap_kinds_interpreted(tmp_path):
    results = {r.trap_id: r.passed for r in check_traps(make_run(tmp_path), golden())}
    assert results == {"T5": True, "T9": True, "T3": True, "T2": True, "T10": True, "T6": False}


@pytest.mark.parametrize("gold, got", [
    ("Срок действия договора до 31 декабря 2026 года", "Срок действия договора до 31 декабря 2025 года"),
    ("арендная плата 120 000 рублей в месяц", "арендная плата 100 000 рублей в месяц"),
    ("до 31.12.2026", "до 31.12.2025; до 31.12.2026"),
    ("оплата ежемесячно до 10 числа", "оплата ежеквартально до 10 числа"),
    ("Арендная плата вносится ежемесячно, не позднее 10 числа текущего месяца, на расчетный счет Арендодателя",
     "Арендная плата вносится ежеквартально, не позднее 10 числа текущего месяца, на расчетный счет Арендодателя"),
    ("в течение 10 рабочих дней после подписания акта", "в течение 10 календарных дней после подписания акта"),
    ("уведомление не менее чем за 2 месяца", "уведомление не менее чем за 2 недели"),
    ("штраф 2 000 руб. уплачивает Дилер", "штраф 2 000 руб. уплачивает Компания"),
    ("по Дополнительному соглашению № D990000002-01", "по Дополнительному соглашению № D990000003-01"),
], ids=["year", "amount", "two_versions", "monthly_quarterly", "long_substitution", "day_kind", "time_unit",
        "party", "doc_number"])
def test_similar_text_tells_different_terms_apart(gold, got):
    assert not similar_text(gold, got)


@pytest.mark.parametrize("gold, got", [
    ("до 31.12.2026 включительно", "Договор действует до 31 декабря 2026 года включительно"),
    ("120 000 руб. в месяц без НДС", "120 000 (сто двадцать тысяч) рублей 00 копеек в месяц без НДС"),
    ("ежемесячно, не позднее 10 числа текущего месяца",
     "Арендная плата вносится ежемесячно не позднее 10 (десятого) числа текущего месяца"),
    ("оплата ежемесячно до 10 числа", "ежемесячная оплата до 10-го числа"),
    ("пени 0,1% за каждый день просрочки", "пени в размере 0.1 % от суммы задолженности за каждый день просрочки"),
    ("до 31.12.2026 с пролонгацией", "действует до 31.12.2026 с автоматической пролонгацией"),
    ("с 2027-01-01 по 31.12.2027", "с 01.01.2027 по «31» декабря 2027 г."),
    ("штраф 5000 руб.", "штраф 5 000 руб."),
    ("ежемесячно в течение 10 рабочих дней после подписания акта",
     "2.2. Оплата производится ежемесячно в течение 10 (десяти) рабочих дней после подписания Сторонами акта."),
    ("односторонний отказ любой из Сторон с письменным уведомлением не менее чем за 2 месяца",
     "Каждая из Сторон вправе отказаться от Договора, письменно уведомив другую Сторону не менее чем за 2 (два) месяца"),
], ids=["date_forms", "kopecks_and_words", "longer", "word_forms", "percent", "extra_words", "iso_date", "nbsp",
        "clause_number", "reworded"])
def test_similar_text_tolerates_paraphrase(gold, got):
    assert similar_text(gold, got)


def _set(attr, value, i=0, field="ops"):
    def tweak(records):
        setattr(getattr(records["s"], field)[i], attr, value)
    return tweak


def test_op_with_wrong_date_or_target_is_wrong(tmp_path):
    assert score_run(make_run(tmp_path / "ok"), golden()).critical_accepted_wrong == 0
    late = score_run(make_run(tmp_path / "date", tweak=_set("effective_from", date(2026, 9, 1))), golden())
    elsewhere = score_run(make_run(tmp_path / "target", tweak=_set("target", "п. 4.2")), golden())
    assert late.critical_accepted_wrong == elsewhere.critical_accepted_wrong == 1


def test_op_date_defaults_to_the_document_date_on_both_sides(tmp_path):
    gs = golden()
    gs.documents[1].ops[0].effective_from = None
    run = make_run(tmp_path, tweak=_set("effective_from", date(2026, 7, 1)))
    assert score_run(run, gs).critical_accepted_wrong == 0


def test_rate_with_wrong_vat_is_wrong(tmp_path):
    s = score_run(make_run(tmp_path, tweak=_set("vat", "included", field="rates")), golden())
    assert s.critical_accepted_wrong == 1


@pytest.mark.parametrize("a, b", [
    ("п. 12.2", "пункт 12.2"), ("12.2", "Пункт 12.2. Договора"), ("подпункт 1.1.1", "пп. 1.1.1"),
    ("Приложение № 1, п. 1.6", "п. 1.6 Приложения 1"), ("Приложение 7", "приложение №7"), ("1-8", "разделы 1–8"),
    ("D990000003-01", "Дополнительное соглашение № D990000003-01 от «02» февраля 2026 г."), (None, "весь договор"),
])
def test_clause_references_normalize_to_one_form(a, b):
    assert normalize_target(a) == normalize_target(b)


@pytest.mark.parametrize("a, b", [("п. 12.2", "п. 12.3"), ("Приложение № 1, п. 1.6", "Приложение № 2, п. 1.6"),
                                  ("D990000003-01", "D990000004-01"), ("п. 3.1", None)])
def test_different_clause_references_stay_different(a, b):
    assert normalize_target(a) != normalize_target(b)


def test_contract_the_run_did_not_fold_is_not_refolded_by_code(tmp_path):
    import json

    run = make_run(tmp_path)
    (run / "kb" / "consolidated.json").write_text("{}")  # e.g. the single-call variant skipped this contract
    (run / "metrics.json").write_text(json.dumps({"as_of": AS_OF.isoformat()}))
    assert score_run(run, golden()).consolidation_acc == 0.0
    assert {t.trap_id: t.passed for t in check_traps(run, golden())}["T5"] is False


def test_latest_full_run_skips_partial_and_unfinished_runs(tmp_path):
    import json

    from contract_analyst.eval.metrics import latest_full_run

    def run(name, scope=None, finished=True):
        (tmp_path / name / "kb").mkdir(parents=True)
        if finished:
            (tmp_path / name / "metrics.json").write_text(json.dumps({} if scope is None else {"scope": scope}))
        return tmp_path / name

    full = {"chain": None, "doc": None}
    run("20261030-090000-B", full)
    newest_full = run("20261030-100000-B", full)
    run("20261030-110000-B", {"chain": MAIN, "doc": None})
    run("20261030-120000-B", {"chain": None, "doc": "ДС1.pdf"})
    run("20261030-130000-B", full, finished=False)  # still running: metrics.json is written last
    run("20261030-140000-B")  # written before runs recorded their scope
    assert latest_full_run(tmp_path) == newest_full
    assert latest_full_run(tmp_path / "missing") is None


def test_values_the_fold_did_not_accept_score_nothing(tmp_path):
    from pydantic import TypeAdapter

    from contract_analyst.models import ConsolidatedContract
    from tests.fixtures.builders import acc

    run = make_run(tmp_path, tweak=lambda records: records["s"].terms.update(validity=acc("до 31.12.2026", "s")))
    contract = KnowledgeBase(run / "kb").consolidated()[MAIN]
    contract.terms["validity"].current.status = "conflict"
    for row in [*contract.rates, *contract.planned_rates]:
        row.status = "conflict"
    adapter = TypeAdapter(dict[str, ConsolidatedContract])
    (run / "kb" / "consolidated.json").write_bytes(adapter.dump_json({MAIN: contract}))
    gs = golden()
    gs.chains[0].terms_at = {AS_OF: {"validity": "до 31.12.2026"}}
    gs.traps.append(GoldenTrap(trap_id="T7", kind="term_value_at", contract_number=MAIN,
                               expected={"as_of": "2026-10-30", "term": "validity", "value": "до 31.12.2026"}))
    assert score_run(run, gs).consolidation_acc == 1 / 3  # only the status is right
    traps = {t.trap_id: t.passed for t in check_traps(run, gs)}
    assert traps["T5"] and not traps["T9"] and not traps["T7"]


def test_truncated_answers_lower_the_json_validity_rate(tmp_path):
    from contract_analyst.models import Issue

    def truncated(records):
        records["s"].issues.append(Issue(code="llm_truncated", message="s.pdf: группа money — ответ обрезан"))

    run = make_run(tmp_path, tweak=truncated)
    (run / "metrics.json").write_text('{"llm_usage": {"calls": 4}}', encoding="utf-8")
    assert score_run(run, golden()).json_valid_rate == 0.75


def test_rate_without_its_own_start_runs_from_its_document(tmp_path):
    def undated(records):
        records["s"].rates[0] = records["s"].rates[0].model_copy(update={"valid_from": None})

    baseline = score_run(make_run(tmp_path / "dated"), golden())
    s = score_run(make_run(tmp_path / "undated", tweak=undated), golden())
    assert s.critical_accepted_wrong == 0 and s.critical_accepted_correct == baseline.critical_accepted_correct
    assert s.consolidation_acc == baseline.consolidation_acc


# --- where consolidation points are lost --------------------------------------------------------------------------

_UNCHANGED = {"doc_kind_acc": 1.0, "chain_acc": 1.0, "critical_total": 8, "clusters_match": True, "traps_passed": 5,
              "traps_total": 6, "wall_seconds": 0.0, "llm_calls": 0, "input_tokens": 0, "output_tokens": 0,
              "json_valid_rate": 1.0, "tool_call_valid_rate": 1.0, "safety_net": []}


def _mixed(records):
    """The supplement's first rate lost its end date; its validity glues the old value to the new one."""
    records["s"].rates[0] = records["s"].rates[0].model_copy(update={"valid_to": None})
    records["s"].terms["validity"] = acc("до 31.12.2025; до 31.12.2026", "s")


def _mixed_golden():
    gs = golden()
    gs.chains[0].terms_at = {AS_OF: {"validity": "до 31.12.2026", "payment_terms": "ежемесячно до 10 числа"}}
    return gs


@pytest.mark.parametrize("tweak, gs, changed", [
    (None, golden, {"critical_accepted_correct": 8, "critical_accepted_wrong": 0, "critical_coverage": 1.0,
                    "consolidation_acc": 1.0}),
    (_set("vat", "included", field="rates"), golden, {"critical_accepted_correct": 7, "critical_accepted_wrong": 1,
                                                      "critical_coverage": 0.875, "consolidation_acc": 0.5}),
    (_mixed, _mixed_golden, {"critical_accepted_correct": 7, "critical_accepted_wrong": 1, "critical_coverage": 0.875,
                             "consolidation_acc": 0.25}),
], ids=["perfect", "wrong_vat", "mixed"])
def test_existing_scores_are_unchanged(tmp_path, tweak, gs, changed):
    """Pinned from the code before the miss breakdown existed (d66fc23): explaining the scores must not move them."""
    s = score_run(make_run(tmp_path, tweak=tweak), gs())
    assert s.model_dump(include=set(_UNCHANGED) | set(changed)) == _UNCHANGED | changed


def _contract(status="active", rates=(), terms=None):
    """A folded contract as a run stores it; a rate is (amount, vat, valid_from, valid_to)."""
    return ConsolidatedContract(
        contract_number=MAIN, as_of=AS_OF, status=status,
        rates=[RateRow(label="Ставка", amount=Decimal(a), vat=vat, valid_from=vf, valid_to=vt, status="accepted")
               for a, vat, vf, vt in rates],
        terms={k: TermState(name=k, current=v if isinstance(v, Extracted) else acc(v)) for k, v in (terms or {}).items()})


def _one_chain(status=None, rates=None, terms=None):
    """A golden set of one chain on AS_OF with only the points given."""
    golden_rates = [GoldenRate(label="Ставка", amount=Decimal(a), vat=vat, valid_from=vf, valid_to=vt)
                    for a, vat, vf, vt in rates or ()]
    chain = GoldenChain(contract_number=MAIN, status_at={AS_OF: status} if status else {},
                        rates_at={AS_OF: golden_rates} if rates is not None else {},
                        terms_at={AS_OF: terms} if terms else {})
    return GoldenSet(as_of_dates=[AS_OF], chains=[chain])


def _scored(gs, contract):
    return score_consolidation(gs, lambda number, when: contract, records={})


def test_a_lost_status_point_says_the_status_was_wrong():
    scored = _scored(_one_chain(status="terminated"), _contract("active"))
    assert scored.acc == 0.0
    assert scored.misses == [{"chain": MAIN, "date": "2026-10-30", "point": "status", "reason": "status_wrong",
                              "expected": "terminated", "got": "active"}]
    assert _scored(_one_chain(status="terminated"), None).misses[0]["got"] is None  # the run folded no such contract
    assert _scored(_one_chain(status="active"), _contract("active")).misses == []


_H2, _Y27 = (date(2026, 7, 1), date(2026, 12, 31)), (date(2027, 1, 1), None)


@pytest.mark.parametrize("golden_rows, got_rows, expected", [
    ([("675", "excluded", *_H2), ("1200", "excluded", *_Y27)], [("675", "excluded", *_H2)],
     [("rates_missing", "1200 excluded 2027-01-01–…", None)]),
    ([("675", "excluded", *_H2)], [("675", "excluded", *_H2), ("300", "excluded", None, None)],
     [("rates_extra", None, "300 excluded …–…")]),
    ([("675", "excluded", *_H2)], [("675", "excluded", date(2026, 7, 1), None)],
     [("rates_period", "675 excluded 2026-07-01–2026-12-31", "675 excluded 2026-07-01–…")]),
    ([("675", "excluded", *_H2), ("1200", "excluded", *_Y27)],
     [("675", "excluded", None, None), ("675", "included", *_H2)],
     [("rates_missing", "1200 excluded 2027-01-01–…", None), ("rates_extra", None, "675 included 2026-07-01–2026-12-31"),
      ("rates_period", "675 excluded 2026-07-01–2026-12-31", "675 excluded …–…")]),
], ids=["missing", "extra", "period", "several"])
def test_a_lost_rate_point_says_which_rows_differ(golden_rows, got_rows, expected):
    misses = _scored(_one_chain(rates=golden_rows), _contract(rates=got_rows)).misses
    assert [(m["point"], m["reason"], m["expected"], m["got"]) for m in misses] == [("rates", *e) for e in expected]


def test_a_contract_the_run_did_not_fold_misses_its_whole_rate_set():
    (miss,) = _scored(_one_chain(rates=[("675", "excluded", *_H2)]), None).misses
    assert (miss["reason"], miss["expected"], miss["got"]) == ("rates_missing", "675 excluded 2026-07-01–2026-12-31", None)


def test_rate_rows_show_progress_that_the_all_or_nothing_point_hides():
    later = date(2027, 2, 1)
    gs = _one_chain(rates=[("675", "excluded", *_H2), ("1200", "excluded", *_Y27)])
    gs.as_of_dates.append(later)
    gs.chains[0].rates_at[later] = [GoldenRate(label="Ставка", amount=Decimal(1200), vat="excluded",
                                               valid_from=date(2027, 1, 1))]
    folds = {AS_OF: _contract(rates=[("675", "excluded", *_H2), ("999", "excluded", None, None)]),
             later: _contract(rates=[("1200", "excluded", *_Y27)])}
    scored = score_consolidation(gs, lambda number, when: folds[when], records={})
    assert scored.acc == 0.5  # the first date's rate set is not the golden one
    assert scored.rows_acc == 2 / 3  # yet two of the three golden rows were folded right
    assert _scored(_one_chain(status="active"), _contract()).rows_acc == 1.0  # no rate points


_COMBINED = "по факсу в течение 3 дней; копии через ЭДО в течение 1 дня"  # a golden value that is itself two clauses


@pytest.mark.parametrize("gold, state, reason", [
    ("до 31.12.2026", None, "term_absent"),
    ("до 31.12.2026", Extracted(status="rejected", reason="quote_not_found"), "term_absent"),
    ("до 31.12.2026", Extracted(status="conflict", reason="ops_not_identified"), "term_conflict"),
    ("до 31.12.2026", "до 31.12.2025; до 31.12.2026", "term_glued"),
    ("до 31.12.2026", "до 31.12.2025, с продлением до 31.12.2026", "term_glued"),
    (_COMBINED, _COMBINED + "; оригиналы в течение 5 дней", "term_glued"),
    ("до 31.12.2026", "до 31.12.2025", "term_different"),
    ("оплата ежемесячно до 10 числа", "оплата ежеквартально до 10 числа", "term_different"),
    ("заключен на неопределенный срок", "до 31.12.2026", "term_different"),
    (_COMBINED, "копии через ЭДО в течение 1 дня", "term_different"),
], ids=["no_state", "rejected", "conflict", "old_and_new", "extra_date", "golden_combined_plus_more", "other_date",
        "other_word", "no_golden_numbers", "replaced_instead_of_combined"])
def test_a_lost_term_point_says_why(gold, state, reason):
    terms = {} if state is None else {"validity": state}
    (miss,) = _scored(_one_chain(terms={"validity": gold}), _contract(terms=terms)).misses
    assert (miss["chain"], miss["point"], miss["reason"]) == (MAIN, "terms.validity", reason)


def test_a_term_miss_shows_numbers_and_dates_not_the_wording():
    gs = _one_chain(terms={"rent": "арендная плата 120 000 рублей в месяц до 31.12.2026",
                           "plan": "не применяется с 01.03.2026", "validity": "заключен на неопределенный срок"})
    many = "; ".join(f"ставка {n} 000 рублей с 01.0{n}.2026" for n in range(1, 10))
    contract = _contract(terms={
        "rent": "арендная плата 100 000 (сто тысяч) рублей в месяц до 31 декабря 2026 года",
        "plan": "не менее 40 новых клиентов в месяц; не применяется с 01.03.2026", "validity": many})
    misses = {m["point"]: m for m in _scored(gs, contract).misses}
    assert (misses["terms.rent"]["expected"], misses["terms.rent"]["got"]) == ("2026-12-31 120000", "2026-12-31 100000")
    assert (misses["terms.plan"]["expected"], misses["terms.plan"]["got"]) == ("2026-03-01", "40; 2026-03-01")
    assert misses["terms.validity"]["expected"] == ""  # nothing but words
    assert len(misses["terms.validity"]["got"]) == 80 and misses["terms.validity"]["got"].endswith("…")
    assert not [m for m in misses.values() if re.search("[а-яё]", m["expected"] + m["got"], re.IGNORECASE)]


def test_scores_say_where_the_consolidation_points_were_lost(tmp_path):
    s = score_run(make_run(tmp_path / "mixed", tweak=_mixed), _mixed_golden())
    assert s.consolidation_misses == {"status_wrong": 0, "rates_missing": 0, "rates_extra": 0, "rates_period": 1,
                                      "term_absent": 1, "term_conflict": 0, "term_glued": 1, "term_different": 0}
    assert [(m["date"], m["point"], m["reason"], m["expected"], m["got"]) for m in s.consolidation_miss_list] == [
        ("2026-10-30", "rates", "rates_period", "675 unknown 2026-07-01–2026-12-31", "675 unknown 2026-07-01–…"),
        ("2026-10-30", "terms.validity", "term_glued", "2026-12-31", "2025-12-31; 2026-12-31"),
        ("2026-10-30", "terms.payment_terms", "term_absent", "10", None)]
    assert s.rates_rows_acc == 0.0
    perfect = score_run(make_run(tmp_path / "perfect"), golden())
    assert perfect.rates_rows_acc == 1.0 and perfect.consolidation_miss_list == []
    assert set(perfect.consolidation_misses.values()) == {0}
