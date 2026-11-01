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
from contract_analyst.eval.metrics import normalize_target, same_value, score_run, similar_text
from contract_analyst.eval.traps import check_traps
from contract_analyst.kb.store import KnowledgeBase
from contract_analyst.models import OpType
from tests.fixtures.builders import make_record as rec
from tests.fixtures.builders import text_doc

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
