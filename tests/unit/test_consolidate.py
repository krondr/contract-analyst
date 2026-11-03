from datetime import date
from decimal import Decimal

import pytest

from contract_analyst.assemble.chains import build_chains
from contract_analyst.assemble.consolidate import consolidate, covers
from contract_analyst.models import Extracted, Issue, OpType, RateRow, Source, ValueStatus
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


RATES = [("Пакет Макс", 685, date(2026, 7, 1), date(2026, 12, 31)), ("Пакет Макс", 1250, date(2027, 1, 1), None)]


def test_future_rate_is_planned():
    c = run([main(), rec("s", "D990000002-01", parent=MAIN, effective=date(2026, 7, 1), rates=RATES)])
    assert [r.amount for r in c.rates] == [Decimal(685)] and [r.amount for r in c.planned_rates] == [Decimal(1250)]


def test_after_change_date_new_rate_current():
    c = run([main(), rec("s", "D990000002-01", parent=MAIN, effective=date(2026, 7, 1), rates=RATES)], date(2027, 2, 1))
    assert [r.amount for r in c.rates] == [Decimal(1250)] and c.planned_rates == []


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
    c = run([main(rates=[("Пакет Макс", 500, None, None)]), s], date(2027, 2, 1))
    assert [r.amount for r in c.rates] == [Decimal(1250)]


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


@pytest.mark.parametrize("code", ["llm_timeout", "llm_invalid_json", "llm_truncated"])
def test_ops_group_without_a_usable_answer_makes_the_change_a_conflict(code):
    sup = rec("s", "D990000002-01", parent=MAIN, effective=date(2025, 1, 1),
              terms={"payment_terms": ("ежеквартально до 15 числа", "5.1")})
    sup.issues.append(Issue(code=code, message="s.pdf: группа ops — ответа нет"))
    c = run([main(terms={"payment_terms": ("ежемесячно до 10 числа", "5.1")}), sup])
    assert c.terms["payment_terms"].current.status == ValueStatus.CONFLICT


def refused_start(doc_id):
    return Extracted(value=date(2027, 1, 1), status=ValueStatus.REJECTED, reason="date_role: не дата начала",
                     source=Source(doc_id=doc_id, page=1, quote="вступает в силу с 01.01.2027"))


def test_refused_entry_into_force_makes_the_changes_conflicts_not_dated_by_signing():
    sup = rec("s", "D990000002-01", parent=MAIN, doc_date=date(2026, 10, 15), ops=[(OpType.REPLACE_CLAUSE, "5.1", None)],
              terms={"payment_terms": ("ежеквартально", "5.1")},
              rates=[("Аренда", 150000, None, None), ("Уборка", 500, date(2025, 1, 1), None)])
    sup.effective_from = refused_start("s")
    c = run([main(rates=[("Аренда", 100000, None, None)], terms={"payment_terms": ("ежемесячно", "5.1")}), sup])
    assert [(r.label, r.amount) for r in c.rates] == [("Уборка", Decimal(500))] and c.planned_rates == []
    assert c.terms["payment_terms"].current.status == ValueStatus.CONFLICT
    assert any(i.code == "effective_date_unconfirmed" for i in c.issues)


def test_termination_without_a_confirmed_date_leaves_the_status_unknown():
    sup = rec("s", "D990000002-01", parent=MAIN, doc_date=date(2026, 10, 15), ops=[(OpType.TERMINATE, None, None)])
    sup.effective_from = refused_start("s")
    assert run([main(), sup]).status == "unknown"


def test_supplement_that_states_no_start_applies_from_its_signing_date():
    sup = rec("s", "D990000002-01", parent=MAIN, doc_date=date(2026, 10, 15), rates=[("Аренда", 150000, None, None)])
    assert [r.amount for r in run([main(rates=[("Аренда", 100000, None, None)]), sup]).rates] == [Decimal(150000)]


def rate_in(doc_id, label, amount, clause, valid_from=None):
    return RateRow(label=label, amount=Decimal(amount), valid_from=valid_from, status=ValueStatus.ACCEPTED,
                   source=Source(doc_id=doc_id, page=1, quote="цитата", clause_id=clause))


def test_new_wording_of_an_appendix_retires_its_old_rates():
    m = main()
    m.rates = [rate_in("m", "Тариф А", 300, "Приложение 7/1"), rate_in("m", "Тариф Б", 400, "Приложение 7/1"),
               rate_in("m", "Тариф В", 900, "Приложение 7/1", date(2027, 1, 1)), rate_in("m", "Ставка", 50, "4.2")]
    sup = rec("s", "D990000002-01", parent=MAIN, effective=date(2025, 1, 1),
              ops=[(OpType.REPLACE_APPENDIX, "Приложение № 7", None)])
    sup.rates = [rate_in("s", "Тариф А", 500, "Приложение 7/1")]
    c = run([m, sup])
    assert [(r.label, r.amount) for r in c.rates] == [("Ставка", Decimal(50)), ("Тариф А", Decimal(500))]
    assert c.planned_rates == []


def test_rate_of_a_rewritten_clause_ends_when_the_new_wording_starts():
    m = main()
    m.rates = [rate_in("m", "Ставка", 700, "3.1")]
    sup = rec("s", "D990000002-01", parent=MAIN, effective=date(2027, 1, 1), ops=[(OpType.REPLACE_CLAUSE, "3.1", None)])
    assert [r.amount for r in run([m, sup]).rates] == [Decimal(700)]
    assert run([m, sup], date(2027, 2, 1)).rates == []


def test_current_term_knows_when_it_started():
    c = run([main(terms={"validity": "до 31.12.2025"}),
             rec("s", "D990000002-01", parent=MAIN, effective=date(2025, 10, 1), terms={"validity": "до 31.12.2026"},
                 ops=[(OpType.REPLACE_CLAUSE, "7.1", None)])])
    assert c.terms["validity"].since == date(2025, 10, 1)
    assert [h.effective_from for h in c.terms["validity"].history] == [date(2022, 8, 23)]


def test_terminating_a_supplement_cancels_it_not_the_contract():
    c = run([main(terms={"payment_terms": ("оплата в течение 10 дней", "5.1")}),
             rec("s2", "D990000002-01", parent=MAIN, effective=date(2025, 1, 1),
                 terms={"payment_terms": "оплата в течение 20 дней"}, ops=[(OpType.REPLACE_CLAUSE, "5.1", None)]),
             rec("s3", "D990000003-01", parent=MAIN, effective=date(2026, 5, 1),
                 ops=[(OpType.TERMINATE, "Дополнительное соглашение № D990000002-01", date(2026, 5, 1))])])
    assert c.status == "active"
    assert c.terms["payment_terms"].current.value == "оплата в течение 10 дней"
    assert [e.invalidated for e in c.timeline if e.doc_id == "s2"] == [True]


@pytest.mark.parametrize("op", [OpType.TERMINATE, OpType.SUSPEND_PART])
def test_ending_a_document_outside_the_chain_leaves_the_status(op):
    c = run([main(), rec("s", "D990000002-01", parent=MAIN, effective=date(2026, 5, 1),
                         ops=[(op, "Договор № D980000007-01", date(2026, 5, 1))])])
    assert c.status == "active"


def shorter_term(**kw):
    """A supplement whose payment term differs from the contract's «130 дней …» yet is a substring of it."""
    return rec("s", "D990000002-01", parent=MAIN, effective=date(2024, 1, 1),
               terms={"payment_terms": "30 дней с даты поставки"}, **kw)


def test_combined_value_keeps_a_new_value_found_inside_the_current_one():
    c = run([main(terms={"payment_terms": "130 дней с даты поставки"}), shorter_term()])
    assert c.terms["payment_terms"].current.value == "130 дней с даты поставки; 30 дней с даты поставки"


def test_value_found_inside_the_current_one_is_a_conflict_when_ops_are_unread():
    s = shorter_term()
    s.issues.append(Issue(code="llm_invalid_json", message="s.pdf: группа ops — ответ не разобран"))
    current = run([main(terms={"payment_terms": "130 дней с даты поставки"}), s]).terms["payment_terms"].current
    assert current.status == "conflict" and current.reason.startswith("ops_not_identified")


@pytest.mark.parametrize("repeat", ["Оплата  в течение 30 дней.", "аванс 30%", "оплата в течение 30 дней; аванс 30%"])
def test_repeated_value_is_not_added_again(repeat):
    c = run([main(terms={"payment_terms": "оплата в течение 30 дней"}),
             rec("s1", "D990000002-01", parent=MAIN, effective=date(2024, 1, 1), terms={"payment_terms": "аванс 30%"}),
             rec("s2", "D990000003-01", parent=MAIN, effective=date(2025, 1, 1), terms={"payment_terms": repeat})])
    assert c.terms["payment_terms"].current.value == "оплата в течение 30 дней; аванс 30%"

