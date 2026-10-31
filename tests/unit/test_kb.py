import json
from datetime import date
from decimal import Decimal

from contract_analyst.assemble.chains import build_chains
from contract_analyst.assemble.consolidate import consolidate
from contract_analyst.kb.lemmatize import lemmatize
from contract_analyst.kb.store import KnowledgeBase
from contract_analyst.models import Extracted, Source, ValueStatus
from tests.fixtures.builders import make_record as rec
from tests.fixtures.builders import text_doc

MAIN = "D990000001-01"
AS_OF = date(2026, 10, 30)


def small_run():
    main_doc = text_doc([("Договор № D990000001-01\n1. Предмет договора: оказание услуг.\n"
                         "3.1. Ставка вознаграждения составляет 500 рублей.")], doc_id="m")
    sup_doc = text_doc(["Дополнительное соглашение № D990000002-01\n1. Изложить п. 3.1 в новой редакции."], doc_id="s")
    secret = Extracted(value="секретное значение", status=ValueStatus.REJECTED,
                       source=Source(doc_id="m", page=1, quote="нет"), reason="quote_not_found")
    main = rec("m", MAIN, kind="main", doc_date=date(2021, 1, 21), terms={"subject": "оказание услуг"},
               rates=[("Ставка", 500, None, None)])
    main.terms["penalties"] = secret
    sup = rec("s", "D990000002-01", parent=MAIN, effective=date(2026, 7, 1),
              rates=[("Ставка", 700, date(2027, 1, 1), None)])
    records = {"m": main, "s": sup}
    chains = build_chains(list(records.values()))
    consolidated = {c.contract_number: consolidate(c, records, AS_OF) for c in chains}
    return {"documents": {"m": main_doc, "s": sup_doc}, "records": records, "chains": chains, "consolidated": consolidated,
                "clusters": [], "decisions": []}


def test_lemmatize_keeps_numbers():
    assert lemmatize("Ставкой вознаграждения D990000001-01 500") == "ставка вознаграждение d990000001-01 500"


def test_roundtrip_and_inflected_search(tmp_path):
    KnowledgeBase(tmp_path).save(**small_run())
    kb = KnowledgeBase(tmp_path)
    (contract,) = kb.list_contracts()
    assert contract["contract_number"] == MAIN and contract["status"] == "active"
    hits = kb.search_clauses("ставкой вознаграждения")
    assert hits[0].clause_id == "3.1" and hits[0].contract_number == MAIN and hits[0].page == 1


def test_search_can_filter_by_contract(tmp_path):
    kb = KnowledgeBase(tmp_path)
    kb.save(**small_run())
    assert kb.search_clauses("изложить", contract="D990000777-01") == []


def test_rejected_value_not_stored_as_fact(tmp_path):
    KnowledgeBase(tmp_path).save(**small_run())
    assert "секретное значение" not in (tmp_path / "records.json").read_text(encoding="utf-8")


def test_terms_recomputed_for_other_date(tmp_path):
    kb = KnowledgeBase(tmp_path)
    kb.save(**small_run())
    assert [r.amount for r in kb.get_contract_terms(MAIN).rates] == [Decimal(500)]
    assert [r.amount for r in kb.get_contract_terms(MAIN, date(2027, 2, 1)).rates] == [Decimal(700)]


def test_valid_refs_and_issues(tmp_path):
    kb = KnowledgeBase(tmp_path)
    kb.save(**small_run())
    assert "doc:m:p1" in kb.valid_refs() and "doc:m:p2" not in kb.valid_refs()
    assert isinstance(kb.get_issues(MAIN), list)
    assert json.loads((tmp_path / "consolidated.json").read_text(encoding="utf-8"))[MAIN]["as_of"] == "2026-10-30"
