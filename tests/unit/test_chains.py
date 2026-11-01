from datetime import date

from contract_analyst.assemble.chains import build_chains
from contract_analyst.models import OpType
from tests.fixtures.builders import make_record as rec

MAIN = "D990000001-01"


def codes(chain):
    return [i.code for i in chain.issues]


def test_supplements_link_by_parent_number():
    chains = build_chains([rec("s2", "D990000003-01", parent=MAIN, effective=date(2022, 3, 1)),
                           rec("m", MAIN, kind="main", doc_date=date(2021, 1, 21)),
                           rec("s1", "D990000002-01", parent=MAIN, effective=date(2021, 6, 1))])
    (chain,) = chains
    assert chain.contract_number == MAIN and chain.main_doc_id == "m"
    assert [d.doc_id for d in chain.docs] == ["m", "s1", "s2"] and chain.docs[0].role == "main"


def test_missing_main():
    (chain,) = build_chains([rec("s1", "D990000002-01", parent=MAIN, effective=date(2021, 6, 1))])
    assert chain.main_doc_id is None and "missing_main" in codes(chain)


def test_invalidated_marked():
    chains = build_chains([rec("m", MAIN, kind="main", doc_date=date(2022, 8, 23)),
                           rec("a", "D990000004-01", parent=MAIN, doc_date=date(2024, 2, 14)),
                           rec("b", "D990000005-01", parent=MAIN, doc_date=date(2024, 3, 1),
                               ops=[(OpType.INVALIDATE_DOCUMENT, "D990000004-01", None)])])
    docs = {d.doc_id: d for d in chains[0].docs}
    assert docs["a"].invalidated_by == "b" and docs["b"].invalidated_by is None
    assert "invalidated" in codes(chains[0])


def test_order_by_effective_date_not_file_name():
    (chain,) = build_chains([rec("m", MAIN, kind="main", doc_date=date(2021, 1, 25)),
                             rec("ds3", "D990000006-01", parent=MAIN, effective=date(2021, 10, 21), file_name="ДС3.pdf"),
                             rec("ds4", "D990000007-01", parent=MAIN, effective=date(2021, 7, 7), file_name="ДС4.pdf")])
    assert [d.doc_id for d in chain.docs] == ["m", "ds4", "ds3"]
    assert chain.docs[1].order_basis == "effective_from"


def test_parent_date_conflict():
    (chain,) = build_chains([rec("m", MAIN, kind="main", doc_date=date(2021, 1, 25)),
                             rec("s", "D990000002-01", parent=MAIN, parent_date=date(2021, 1, 21))])
    assert "parent_date_conflict" in codes(chain)
    assert chain.main_date.status == "conflict" and chain.main_date.value == date(2021, 1, 25)
    assert chain.main_date.alternatives


def test_main_date_from_references():
    (chain,) = build_chains([rec("m", MAIN, kind="main", rejected_date=date(2022, 8, 28)),
                             rec("a", "D990000002-01", parent=MAIN, parent_date=date(2022, 8, 23)),
                             rec("b", "D990000003-01", parent=MAIN, parent_date=date(2022, 8, 23))])
    assert chain.main_date.status == "accepted" and chain.main_date.value == date(2022, 8, 23)
    assert chain.main_date.reason == "inferred_from_references" and "date_from_references" in codes(chain)


def test_order_by_number_fallback():
    (chain,) = build_chains([rec("m", MAIN, kind="main"),
                             rec("b", "D240000002-01", parent=MAIN), rec("a", "D230000001-01", parent=MAIN)])
    assert [d.doc_id for d in chain.docs] == ["m", "a", "b"]
    assert chain.docs[1].order_basis == "number" and "order_by_number" in codes(chain)


def test_target_missing():
    (chain,) = build_chains([rec("m", MAIN, kind="main", doc_date=date(2020, 8, 19)),
                             rec("s", "D990000002-01", parent=MAIN, ops=[(OpType.RESTATE_SECTION, "D990000999-01", None)])])
    assert "target_missing" in codes(chain)


def test_party_name_mismatch():
    (chain,) = build_chains([rec("m", MAIN, kind="main", party_names=["ООО «Ромашка»"]),
                             rec("s", "D990000002-01", parent=MAIN, party_names=["ООО «Лютик»"])])
    assert "party_name_mismatch" in codes(chain)


def test_same_party_written_differently_is_not_mismatch():
    (chain,) = build_chains([rec("m", MAIN, kind="main", party_names=["ООО «Ромашка»"]),
                             rec("s", "D990000002-01", parent=MAIN, party_names=['ООО "Ромашка"'])])
    assert "party_name_mismatch" not in codes(chain)


def test_self_referencing_document_is_the_main():
    (chain,) = build_chains([rec("m", MAIN, kind="supplement", parent=MAIN, doc_date=date(2021, 1, 25)),
                             rec("s", "D990000002-01", parent=MAIN, effective=date(2021, 7, 7))])
    assert chain.main_doc_id == "m" and chain.docs[0].role == "main"


def test_draft_main_flagged():
    (chain,) = build_chains([rec("m", MAIN, kind="main", is_draft=True)])
    assert "draft_main" in codes(chain)


def test_supplement_amending_another_supplement_joins_the_root_chain():
    (chain,) = build_chains([rec("m", MAIN, kind="main", doc_date=date(2021, 1, 21)),
                             rec("s1", "D990000002-01", parent=MAIN, effective=date(2024, 1, 1)),
                             rec("s2", "D990000003-01", parent="D990000002-01", effective=date(2024, 6, 1))])
    assert [d.doc_id for d in chain.docs] == ["m", "s1", "s2"]
    assert "amends_supplement" in codes(chain) and "missing_main" not in codes(chain)


def test_duplicate_main_prefers_the_earliest_signed_original():
    original = rec("zz", MAIN, kind="main", doc_date=date(2021, 1, 21))
    restated = rec("aa", MAIN, kind="main", doc_date=date(2024, 3, 1))
    (chain,) = build_chains([original, restated])
    assert chain.main_doc_id == "zz" and [d.doc_id for d in chain.docs] == ["zz", "aa"]


def test_same_effective_date_orders_by_signing_date_before_number():
    a = rec("a", "D990000002-01", parent=MAIN, doc_date=date(2024, 3, 1), effective=date(2024, 4, 1))
    b = rec("b", "D990000009-01", parent=MAIN, doc_date=date(2024, 2, 1), effective=date(2024, 4, 1))
    (chain,) = build_chains([rec("m", MAIN, kind="main", doc_date=date(2021, 1, 21)), a, b])
    assert [d.doc_id for d in chain.docs] == ["m", "b", "a"]
