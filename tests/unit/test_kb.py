import json
from datetime import date
from decimal import Decimal

from contract_analyst.agent.steps import unverified_in_kb
from contract_analyst.assemble.chains import build_chains
from contract_analyst.assemble.consolidate import consolidate
from contract_analyst.kb.lemmatize import lemmatize
from contract_analyst.kb.store import KnowledgeBase
from contract_analyst.models import AmendmentOp, Extracted, OpType, Party, Source, ValueStatus
from contract_analyst.report.verify import verify_text
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


def test_no_rejected_value_reaches_the_knowledge_base(tmp_path):
    run = small_run()
    sup = run["records"]["s"]
    sup.effective_to = Extracted(value=date(2027, 12, 31), status=ValueStatus.REJECTED,
                                 reason="bad_format: дата 2027-12-31 вне диапазона")
    sup.parties = [Party(role="p0", org_type=Extracted(status=ValueStatus.ABSENT), name=Extracted(
        value="ООО «Тайна»", status=ValueStatus.REJECTED, source=Source(doc_id="s", page=1, quote="ООО «Тайна»"),
        reason="value_not_in_quote: ООО «Тайна»"))]
    KnowledgeBase(tmp_path).save(**run)
    stored = (tmp_path / "records.json").read_text(encoding="utf-8")
    assert "2027-12-31" not in stored and "Тайна" not in stored and "секретное значение" not in stored
    assert KnowledgeBase(tmp_path).records()["s"].effective_to.status == ValueStatus.REJECTED


def test_terms_recomputed_for_other_date(tmp_path):
    kb = KnowledgeBase(tmp_path)
    kb.save(**small_run())
    assert [r.amount for r in kb.get_contract_terms(MAIN).rates] == [Decimal(500)]
    assert [r.amount for r in kb.get_contract_terms(MAIN, date(2027, 2, 1)).rates] == [Decimal(700)]


def test_sources_back_claims_with_page_text_and_cited_values(tmp_path):
    kb = KnowledgeBase(tmp_path)
    kb.save(**small_run())
    sources = kb.sources()
    assert set(sources) == kb.valid_refs()
    for text in ["Ставка вознаграждения 500 рублей [doc:m:p1].",  # page text
                 "С 01.01.2027 ставка 700 руб. [doc:s:p1]."]:  # a rate cited with that page
        assert verify_text(text, sources) == (text, 0), text
    assert verify_text("Ставка 700 руб. [doc:m:p1].", sources) == ("", 1)


def test_valid_refs_and_issues(tmp_path):
    kb = KnowledgeBase(tmp_path)
    kb.save(**small_run())
    assert "doc:m:p1" in kb.valid_refs() and "doc:m:p2" not in kb.valid_refs()
    assert isinstance(kb.get_issues(MAIN), list)
    assert json.loads((tmp_path / "consolidated.json").read_text(encoding="utf-8"))[MAIN]["as_of"] == "2026-10-30"


def saved(tmp_path, records):
    chains = build_chains(list(records.values()))
    kb = KnowledgeBase(tmp_path / "kb")
    kb.save(documents={}, records=records, chains=chains,
            consolidated={c.contract_number: consolidate(c, records, AS_OF) for c in chains}, clusters=[], decisions=[])
    return kb


def test_refold_at_another_date_still_sees_refused_operations(tmp_path):
    main = rec("m", MAIN, kind="main", doc_date=date(2024, 1, 10),
               terms={"payment_terms": ("ежемесячно до 10 числа", "5.1")})
    sup = rec("s", "D990000002-01", parent=MAIN, effective=date(2025, 1, 1),
              terms={"payment_terms": ("ежеквартально до 15 числа", "5.1")})
    sup.ops.append(AmendmentOp(op=OpType.REPLACE_CLAUSE, target="5.1", status=ValueStatus.REJECTED,
                               reason="not_operative: нет глагола", source=Source(doc_id="s", page=1, quote="цитата")))
    kb = saved(tmp_path, {"m": main, "s": sup})
    assert kb.get_contract_terms(MAIN, date(2027, 2, 1)).terms["payment_terms"].current.status == ValueStatus.CONFLICT
    assert unverified_in_kb(tmp_path / "kb") == []


def test_term_history_dates_the_current_value(tmp_path):
    main = rec("m", MAIN, kind="main", doc_date=date(2021, 1, 21), terms={"validity": "до 31.12.2025"})
    sup = rec("s", "D990000002-01", parent=MAIN, effective=date(2026, 7, 1), terms={"validity": "до 31.12.2026"},
              ops=[(OpType.REPLACE_CLAUSE, "7.1", None)])
    history = saved(tmp_path, {"m": main, "s": sup}).get_term_history(MAIN, "validity")
    assert [(h.value, h.effective_from) for h in history] == [("до 31.12.2025", date(2021, 1, 21)),
                                                              ("до 31.12.2026", date(2026, 7, 1))]

