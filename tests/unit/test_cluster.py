from datetime import date
from decimal import Decimal

from contract_analyst.analyze.cluster import (
    cluster_contracts,
    compare_cluster,
    compute_facets,
    summarize_cluster,
)
from contract_analyst.assemble.chains import build_chains
from contract_analyst.assemble.consolidate import consolidate
from contract_analyst.config import load_settings
from contract_analyst.llm.scripted import ScriptedClient
from contract_analyst.models import Extracted, Party, ValueStatus
from tests.fixtures.builders import acc
from tests.fixtures.builders import make_record as rec

AS_OF = date(2026, 10, 30)


def _setup(records):
    chains = build_chains(records)
    by_id = {r.doc_id: r for r in records}
    contracts = {c.contract_number: consolidate(c, by_id, AS_OF) for c in chains}
    return chains, by_id, contracts


def _with_city(record, city, org="ООО"):
    return record.model_copy(update={"city": acc(city, record.doc_id),
                                     "parties": [Party(role="агент", org_type=acc(org, record.doc_id),
                                                       name=acc("Ромашка", record.doc_id))]})


def test_same_category_one_cluster_with_differences():
    a = _with_city(rec("a", "D990000001-01", kind="main", doc_date=date(2021, 1, 1),
                       category="commercial_representation", rates=[("Тариф А", 500, None, None)]), "Тверь")
    b = _with_city(rec("b", "D990000002-01", kind="main", doc_date=date(2022, 1, 1),
                       category="commercial_representation", rates=[("Тариф А", 700, None, None)]), "Псков", org="ИП")
    chains, by_id, contracts = _setup([a, b])
    (cluster,) = cluster_contracts(list(contracts.values()), by_id, chains, load_settings())
    assert cluster.subject_category == "commercial_representation" and cluster.label == "Коммерческое представительство"
    assert sorted(cluster.members) == ["D990000001-01", "D990000002-01"]
    assert cluster.facets["D990000002-01"]["counterparty_type"] == "ИП"
    terms = {d.term: d.values for d in compare_cluster(cluster, contracts)}
    assert terms["city"] == {"D990000001-01": "Тверь", "D990000002-01": "Псков"}
    assert terms["rate:Тариф А"] == {"D990000001-01": "500", "D990000002-01": "700"}


def test_category_from_main_else_majority_else_other():
    records = [rec("m", "D990000001-01", kind="main", category="other"),
               rec("s1", "D990000002-01", parent="D990000001-01", category="lease"),
               rec("s2", "D990000003-01", parent="D990000001-01", category="lease"),
               rec("x", "D990000009-01", kind="main", category="services")]
    chains, by_id, contracts = _setup(records)
    by_member = {m: c.subject_category for c in cluster_contracts(list(contracts.values()), by_id, chains, load_settings())
                 for m in c.members}
    assert by_member == {"D990000001-01": "lease", "D990000009-01": "services"}


def test_facets_say_redacted_or_no_data():
    m = rec("m", "D990000001-01", kind="main", doc_date=date(2021, 5, 1)).model_copy(update={"parties": [
        Party(role="агент", org_type=Extracted(status=ValueStatus.REDACTED), name=Extracted(status=ValueStatus.REDACTED))]})
    _chains, by_id, contracts = _setup([m])
    facets = compute_facets(contracts["D990000001-01"], by_id["m"])
    assert facets["counterparty_type"] == "вымарано" and facets["remuneration_model"] == "нет данных"
    assert facets["year"] == "2021" and facets["status"] == "действует"


def test_summary_uses_llm_and_keeps_refs():
    a = rec("a", "D990000001-01", kind="main", category="lease", rates=[("Аренда", 100000, None, None)])
    chains, by_id, contracts = _setup([a])
    (cluster,) = cluster_contracts(list(contracts.values()), by_id, chains, load_settings())
    text = summarize_cluster(cluster, contracts, ScriptedClient(['{"summary": "Ставка 100000 [doc:a:p1]."}']))
    assert text == "Ставка 100000 [doc:a:p1]."
    assert Decimal(100000) == contracts["D990000001-01"].rates[0].amount
