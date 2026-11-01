from datetime import date
from decimal import Decimal

import pytest

from contract_analyst.assemble.chains import build_chains
from contract_analyst.assemble.consolidate import consolidate, covers
from contract_analyst.models import Issue, OpType, ValueStatus
from tests.fixtures.builders import make_record as rec

MAIN = "D990000001-01"
AS_OF = date(2026, 10, 30)


def run(records, as_of=AS_OF):
    (chain,) = build_chains(records)
    return consolidate(chain, {r.doc_id: r for r in records}, as_of)


def main(**kw):
    return rec("m", MAIN, kind="main", doc_date=kw.pop("doc_date", date(2022, 8, 23)), **kw)


def test_replace_updates_current_keeps_history():
    c = run([main(terms={"validity": "до 31.12.2025"}),
             rec("s", "D990000002-01", parent=MAIN, effective=date(2025, 10, 1), terms={"validity": "до 31.12.2026"},
                 ops=[(OpType.REPLACE_CLAUSE, "7.1", None)])])
    state = c.terms["validity"]
    assert state.current.value == "до 31.12.2026" and [h.value for h in state.history] == ["до 31.12.2025"]


def test_restate_document_resets_base():
    c = run([main(terms={"subject": "старый предмет", "penalties": "штраф 1000"}),
             rec("s", "D990000002-01", parent=MAIN, effective=date(2024, 1, 1),
                 ops=[(OpType.RESTATE_DOCUMENT, MAIN, None)], terms={"subject": "новый предмет"})])
    assert c.terms["subject"].current.value == "новый предмет"
    assert c.terms["penalties"].current.status == "absent" and c.terms["penalties"].history[0].value == "штраф 1000"


def test_invalidated_not_applied_but_in_timeline():
    c = run([main(),
             rec("a", "D990000004-01", parent=MAIN, doc_date=date(2024, 2, 14), terms={"termination": "расторгнут"}),
             rec("b", "D990000005-01", parent=MAIN, doc_date=date(2024, 3, 1),
                 ops=[(OpType.INVALIDATE_DOCUMENT, "D990000004-01", None)])])
    assert "termination" not in c.terms
    assert [e.invalidated for e in c.timeline if e.doc_id == "a"] == [True]


def test_suspend_part_is_not_termination():
    c = run([main(), rec("s", "D990000002-01", parent=MAIN, effective=date(2024, 3, 29),
                         ops=[(OpType.SUSPEND_PART, None, date(2024, 3, 29))])])
    assert (c.status, c.status_since) == ("partially_suspended", date(2024, 3, 29))


def test_terminate_in_future_is_planned():
    records = [main(), rec("s", "D990000002-01", parent=MAIN, effective=date(2026, 9, 1),
                           ops=[(OpType.TERMINATE, None, date(2026, 9, 1))])]
    assert (run(records).status, run(records).status_since) == ("terminated", date(2026, 9, 1))
    early = run(records, date(2026, 8, 1))
    assert early.status == "active" and any("запланировано" in e.summary for e in early.timeline)


RATES = [("Тариф Макс", 675, date(2026, 7, 1), date(2026, 12, 31)), ("Тариф Макс", 1200, date(2027, 1, 1), None)]


def test_future_rate_is_planned():
    c = run([main(), rec("s", "D990000002-01", parent=MAIN, effective=date(2026, 7, 1), rates=RATES)])
    assert [r.amount for r in c.rates] == [Decimal(675)] and [r.amount for r in c.planned_rates] == [Decimal(1200)]


def test_after_change_date_new_rate_current():
    c = run([main(), rec("s", "D990000002-01", parent=MAIN, effective=date(2026, 7, 1), rates=RATES)], date(2027, 2, 1))
    assert [r.amount for r in c.rates] == [Decimal(1200)] and c.planned_rates == []


def test_new_supplement_rate_replaces_same_label():
    c = run([main(rates=[("Тариф А", 500, None, None)]),
             rec("s", "D990000002-01", parent=MAIN, effective=date(2025, 1, 1), rates=[("Тариф А", 700, None, None)])])
    assert [r.amount for r in c.rates] == [Decimal(700)]


def test_retroactive_effective_from():
    c = run([main(), rec("s", "D990000002-01", parent=MAIN, doc_date=date(2026, 9, 15), effective=date(2026, 7, 1),
                         terms={"validity": "до 2027"})], date(2026, 8, 1))
    assert c.terms["validity"].current.value == "до 2027"


def test_missing_main_terms_from_supplements_only():
    c = run([rec("s", "D990000002-01", parent=MAIN, effective=date(2026, 7, 1), terms={"validity": "до 2027"})])
    assert "no_base" in [i.code for i in c.issues] and set(c.terms) == {"validity"}


def test_as_of_before_start_not_yet_in_force():
    assert run([main()], date(2022, 1, 1)).status == "not_yet_in_force"


def test_expired_supplement_is_not_applied_after_its_end():
    records = [main(terms={"plan_indicators": "базовый план"}),
               rec("s", "D990000002-01", parent=MAIN, effective=date(2021, 10, 1), effective_to=date(2021, 10, 31),
                   terms={"plan_indicators": "план на октябрь"})]
    assert run(records).terms["plan_indicators"].current.value == "базовый план"
    assert run(records, date(2021, 10, 15)).terms["plan_indicators"].current.value == "план на октябрь"
    assert any("истёк" in e.summary for e in run(records).timeline)


def test_supplement_that_only_adds_obligations_combines_terms():
    c = run([main(terms={"payment_terms": "оплата ежемесячно"}),
             rec("s", "D990000002-01", parent=MAIN, effective=date(2024, 1, 1),
                 terms={"payment_terms": "сервисные услуги оплачиваются ежеквартально"})])
    current = c.terms["payment_terms"].current
    assert current.value == "оплата ежемесячно; сервисные услуги оплачиваются ежеквартально"
    assert current.alternatives and c.terms["payment_terms"].history == []


@pytest.mark.parametrize(("target", "clause", "expected"), [
    ("12.2", "12.2", True), ("п. 12.2", "12.2.1", True), ("пункт 12", "12.3", True), ("12.2", "12.1", False),
    ("12.2", "Приложение 1/12.2", False), ("Приложение 1, п. 1.6", "Приложение 1/1.6", True),
    ("п. 1.6 Приложения № 1", "Приложение 1/1.6", True), ("Приложение 1, п. 1.6", "1.6", False),
    ("Приложение № 3", "Приложение 3/2.1", True), ("Приложение 3", "Приложение 3", True),
    ("Приложение 3", "3.1", False), ("п. 1-8", "5.2", True), ("разделы 1–8", "9.1", False),
    ("1-8", "Приложение 1/5.2", False), ("«К настоящему Соглашению прилагается»", "5.2", None),
    ("12.2", None, None), ("D990000001-01 п. 5.2", "5.2", True),
])
def test_covers(target, clause, expected):
    assert covers(target, clause) is expected


def test_supplement_amending_another_document_adds_to_the_contract_terms():
    c = run([main(terms={"payment_terms": "оплата в течение 10 рабочих дней"}),
             rec("s", "D990000002-01", parent=MAIN, effective=date(2024, 5, 1),
                 terms={"payment_terms": "вознаграждение за сервисные услуги ежемесячно"},
                 ops=[(OpType.RESTATE_SECTION, "D990000009-01 п. 1-14", None)])])
    assert c.terms["payment_terms"].current.value == ("оплата в течение 10 рабочих дней; "
                                                       "вознаграждение за сервисные услуги ежемесячно")


def test_replaced_clause_replaces_only_the_term_taken_from_it():
    c = run([main(terms={"notification_channel": ("уведомления по факсу", "12.1"),
                         "payment_terms": ("60% от начислений", "Приложение 1/1.6")}),
             rec("s", "D990000002-01", parent=MAIN, effective=date(2023, 11, 1),
                 terms={"notification_channel": "копии по электронной почте", "payment_terms": "65% от начислений"},
                 ops=[(OpType.REPLACE_CLAUSE, "12.2", None), (OpType.REPLACE_CLAUSE, "Приложение 1, п. 1.6", None)])])
    assert c.terms["notification_channel"].current.value == "уведомления по факсу; копии по электронной почте"
    payment = c.terms["payment_terms"]
    assert payment.current.value == "65% от начислений"
    assert [(h.value, h.effective_from) for h in payment.history] == [("60% от начислений", date(2022, 8, 23))]


def test_change_with_unidentified_operations_is_a_conflict():
    s = rec("s", "D990000002-01", parent=MAIN, effective=date(2025, 10, 1), terms={"validity": "до 31.12.2026"})
    s.issues.append(Issue(code="llm_invalid_json", message="s.pdf: группа ops — ответ не разобран"))
    c = run([main(terms={"validity": "до 31.12.2025"}), s])
    current = c.terms["validity"].current
    assert current.status == "conflict" and current.reason.startswith("ops_not_identified")
    assert [a.doc_id for a in current.alternatives] == ["m", "s"]
    assert "ops_not_identified" in [i.code for i in c.issues]


def test_rejected_operation_makes_the_change_a_conflict():
    s = rec("s", "D990000002-01", parent=MAIN, effective=date(2025, 10, 1), terms={"validity": "до 31.12.2026"},
            ops=[(OpType.REPLACE_CLAUSE, "7.1", None)])
    s.ops[0].status = ValueStatus.REJECTED
    assert run([main(terms={"validity": ("до 31.12.2025", "7.1")}), s]).terms["validity"].current.status == "conflict"


def test_operation_end_date_does_not_expire_the_supplement():
    s = rec("s", "D990000002-01", parent=MAIN, effective=date(2026, 7, 1), rates=RATES,
            ops=[(OpType.REPLACE_APPENDIX, "Приложение 7", date(2026, 7, 1))])
    s.ops[0].effective_to = date(2026, 12, 31)
    c = run([main(rates=[("Тариф Макс", 500, None, None)]), s], date(2027, 2, 1))
    assert [r.amount for r in c.rates] == [Decimal(1200)]


def test_rate_rows_apply_in_date_order_whatever_the_extraction_order():
    rows = [("Тариф", 800, date(2027, 1, 1), None), ("Тариф", 500, date(2026, 7, 1), None)]
    c = run([main(), rec("s", "D990000002-01", parent=MAIN, effective=date(2026, 7, 1), rates=rows)], date(2027, 2, 1))
    assert [r.amount for r in c.rates] == [Decimal(800)]


def test_status_unknown_when_no_date_is_known():
    c = run([rec("s", "D990000002-01", parent=MAIN, terms={"validity": "до 2027"})])
    assert c.status == "unknown" and "no_dates" in [i.code for i in c.issues]


def test_as_of_before_every_known_date_is_not_yet_in_force():
    m = rec("m", MAIN, kind="main", rejected_date=date(2016, 5, 5))
    assert run([m, rec("s", "D990000002-01", parent=MAIN, effective=date(2017, 1, 1))], date(2001, 1, 1)).status == \
        "not_yet_in_force"
    assert run([rec("s", "D990000002-01", parent=MAIN, effective=date(2017, 1, 1))], date(2001, 1, 1)).status == \
        "not_yet_in_force"


def test_suspension_of_a_named_clause_replaces_the_term_taken_from_it():
    c = run([main(terms={"plan_indicators": ("не менее 40 новых клиентов в месяц", "1.1.1"),
                         "subject": ("продвижение услуг", "2.1")}),
             rec("s", "D990000002-01", parent=MAIN, effective=date(2026, 2, 16),
                 terms={"plan_indicators": "не применяется с 01.03.2026",
                        "subject": "приостановлено привлечение абонентов"},
                 ops=[(OpType.SUSPEND_PART, "1.1.1", date(2026, 3, 1))])])
    assert c.terms["plan_indicators"].current.value == "не применяется с 01.03.2026"
    assert c.terms["subject"].current.value == "продвижение услуг; приостановлено привлечение абонентов"
