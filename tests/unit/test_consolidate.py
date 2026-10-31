from datetime import date
from decimal import Decimal

from contract_analyst.assemble.chains import build_chains
from contract_analyst.assemble.consolidate import consolidate
from contract_analyst.models import OpType
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
             rec("s", "D990000002-01", parent=MAIN, effective=date(2025, 10, 1), terms={"validity": "до 31.12.2026"})])
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
