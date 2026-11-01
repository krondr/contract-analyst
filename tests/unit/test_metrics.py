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
from contract_analyst.eval.metrics import same_value, score_run, similar_text
from contract_analyst.eval.traps import check_traps
from contract_analyst.kb.store import KnowledgeBase
from contract_analyst.models import OpType
from tests.fixtures.builders import make_record as rec
from tests.fixtures.builders import text_doc

MAIN, SUP = "D990000001-01", "D990000002-01"
AS_OF = date(2026, 10, 30)


def make_run(tmp_path, main_number=MAIN):
    records = {"m": rec("m", main_number, kind="main", doc_date=date(2021, 1, 21), category="lease"),
               "s": rec("s", SUP, parent=MAIN, effective=date(2026, 7, 1),
                        rates=[("Ставка", 675, date(2026, 7, 1), date(2026, 12, 31)), ("Ставка", 1200, date(2027, 1, 1), None)],
                        ops=[(OpType.SUSPEND_PART, None, date(2026, 8, 1))])}
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
