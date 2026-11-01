import json
from datetime import date
from decimal import Decimal

from contract_analyst.assemble.chains import build_chains
from contract_analyst.assemble.merge_llm import consolidate_llm
from contract_analyst.llm.scripted import ScriptedClient
from tests.fixtures.builders import make_record as rec

MAIN = "D990000001-01"


def setup():
    records = {"m": rec("m", MAIN, kind="main", doc_date=date(2024, 1, 10), terms={"validity": "до 31.12.2025"},
                        rates=[("Аренда", 100000, None, None)]),
               "s": rec("s", "D990000002-01", parent=MAIN, effective=date(2025, 4, 1), rates=[("Аренда", 120000, None, None)])}
    (chain,) = build_chains(list(records.values()))
    return chain, records


def test_llm_merge_maps_terms_and_rates_with_sources():
    chain, records = setup()
    answer = {"status": "active", "status_since": None,
              "terms": [{"name": "validity", "value": "до 31.12.2025", "doc_id": "m"},
                        {"name": "penalties", "value": "штраф 1000", "doc_id": "zzz"}],
              "rates": [{"label": "Аренда", "amount": "120000", "doc_id": "s"}], "planned_rates": []}
    client = ScriptedClient([json.dumps(answer, ensure_ascii=False)])
    c = consolidate_llm(chain, records, date(2026, 10, 30), client)
    assert c.status == "active" and [r.amount for r in c.rates] == [Decimal(120000)]
    assert c.terms["validity"].current.status == "accepted" and c.terms["validity"].current.source.doc_id == "m"
    assert c.terms["penalties"].current.status == "conflict"
    assert "Аренда" in client.calls[0]["messages"][1]["content"]


def _merge(records, chain, terms=(), rates=()):
    answer = {"status": "active", "terms": list(terms), "rates": list(rates), "planned_rates": []}
    return consolidate_llm(chain, records, date(2026, 10, 30), ScriptedClient([json.dumps(answer, ensure_ascii=False)]))


def test_merged_value_carries_the_source_of_its_candidate():
    from contract_analyst.models import Source

    chain, records = setup()
    records["m"].terms["validity"].source = Source(doc_id="m", page=2, quote="7.1. Договор действует до 31.12.2025.",
                                                   clause_id="7.1")
    records["s"].rates[0].source = Source(doc_id="s", page=1, quote="арендная плата составляет 120 000 рублей")
    c = _merge(records, chain, terms=[{"name": "validity", "value": "До 31.12.2025.", "doc_id": "s"}],
               rates=[{"label": "аренда", "amount": "120 000", "doc_id": "s"}])
    validity = c.terms["validity"].current
    assert validity.status == "accepted" and validity.value == "до 31.12.2025"
    assert validity.source == records["m"].terms["validity"].source  # the candidate's own page and quote
    assert c.rates[0].status == "accepted" and c.rates[0].source == records["s"].rates[0].source


def test_value_that_is_no_candidate_is_a_conflict():
    chain, records = setup()
    c = _merge(records, chain, terms=[{"name": "validity", "value": "до 31.12.2027", "doc_id": "m"}],
               rates=[{"label": "Аренда", "amount": "130000", "doc_id": "s"}])
    validity, rate = c.terms["validity"].current, c.rates[0]
    assert validity.status == rate.status == "conflict" and validity.source is None and rate.source is None
    assert "не совпадает" in validity.reason and "не совпадает" in rate.reason


def test_llm_merge_falls_back_to_code_on_bad_json():
    chain, records = setup()
    c = consolidate_llm(chain, records, date(2026, 10, 30), ScriptedClient(["x", "y", "z"]))
    assert [r.amount for r in c.rates] == [Decimal(120000)]
    assert "llm_merge_failed" in [i.code for i in c.issues]
