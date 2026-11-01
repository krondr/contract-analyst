from datetime import date
from pathlib import Path

from contract_analyst.analyze.cluster import cluster_contracts
from contract_analyst.assemble.chains import build_chains
from contract_analyst.assemble.consolidate import consolidate
from contract_analyst.config import load_settings
from contract_analyst.models import Decision, Extracted, OpType, Source, ValueStatus
from contract_analyst.report.render import ReflectionItem, ReportContext, render_report
from contract_analyst.report.verify import verify_claims
from tests.fixtures.builders import make_record as rec
from tests.fixtures.builders import text_doc

SNAP = Path(__file__).parent / "snapshots"
MAIN = "D990000001-01"
AS_OF = date(2026, 10, 30)


def small_context() -> ReportContext:
    main = rec("m", MAIN, kind="main", doc_date=date(2021, 1, 21), category="lease",
               terms={"validity": "до 31.12.2025"}, rates=[("Аренда", 100000, None, None)])
    main.terms["penalties"] = Extracted(value="штраф 5000", status=ValueStatus.REJECTED,
                                        source=Source(doc_id="m", page=1, quote="штраф"), reason="quote_not_found: нет")
    sup = rec("s", "D990000002-01", parent=MAIN, effective=date(2025, 4, 1), terms={"validity": "до 31.12.2026"},
              rates=[("Аренда", 120000, None, None)], ops=[(OpType.REPLACE_CLAUSE, "3.1", None)])
    records = {"m": main, "s": sup}
    chains = build_chains(list(records.values()))
    consolidated = {c.contract_number: consolidate(c, records, AS_OF) for c in chains}
    clusters = cluster_contracts(list(consolidated.values()), records, chains, load_settings())
    clusters[0].summary = "Один договор аренды, ставка 120000 рублей [doc:s:p1]."
    documents = [text_doc(["Договор аренды"], doc_id="m", name="Договор.pdf"),
                 text_doc(["Дополнительное соглашение"], doc_id="s", name="ДС1.pdf")]
    return ReportContext(
        run_id="test-run", as_of=AS_OF, model="scripted", prompt_versions={"header": 1}, ocr="tesseract/none",
        documents=documents, records=records, chains=chains, consolidated=consolidated, clusters=clusters,
        decisions=[Decision(issue_code="parent_date_conflict", contract_number=MAIN, decision="оставлены обе даты",
                            rationale="источники противоречат друг другу")],
        reflection=[ReflectionItem(item="Все файлы обработаны", ok=False, note="1 файл не поддерживается")],
        unprocessed=[{"file": "t.xlsx", "reason": "unsupported_format"}], trace_path="trace.jsonl", metrics_path=None)


def test_required_headings_and_refs():
    md = render_report(small_context())
    for h in ["# Отчёт по пакету договоров", "## 1. Сводка запуска", "## 2. Опись документов", "## 3. Кластеры",
              "## 4. Договоры", "### D990000001-01", "#### Цепочка документов", "#### Эволюция условий",
              "#### Действующие условия на 2026-10-30", "## 5. Отказы и противоречия", "## 6. Не обработано",
              "## 7. Reflection", "## 8. Трасса и метрики"]:
        assert h in md, h
    current = md.split("#### Действующие условия")[1].split("##")[0]
    assert "120 000" in current and "[doc:s:p1]" in current and "до 31.12.2026" in current


def test_refusals_decisions_and_unprocessed_listed():
    md = render_report(small_context())
    refusals = md.split("## 5. Отказы и противоречия")[1].split("## 6.")[0]
    assert "штраф 5000" in refusals and "quote_not_found" in refusals and "оставлены обе даты" in refusals
    assert "t.xlsx" in md.split("## 6. Не обработано")[1] and "❌" in md.split("## 7. Reflection")[1]


def test_unreferenced_numeric_sentence_removed():
    md = "<!-- llm -->Ставка выросла до 900 рублей. Условия совпадают [doc:abc:p1].<!-- /llm -->"
    out, removed = verify_claims(md, {"doc:abc:p1"})
    assert "900" not in out and removed == 1 and "Условия совпадают [doc:abc:p1]." in out


def test_invalid_ref_removed():
    out, removed = verify_claims("<!-- llm -->Срок до 2026 года [doc:zzz:p9].<!-- /llm -->", {"doc:abc:p1"})
    assert removed == 1 and "2026" not in out


def test_abbreviations_decimals_and_dates_do_not_split_a_sentence():
    for text in [("120 000 руб. в месяц без НДС, действует с 01.04.2025 по дополнительному соглашению "
                  "D990100002-01 [doc:abc:p1]."),
                 "Ставка по п. 3.1 составляет 700 руб. без НДС [doc:abc:p1].",
                 "Сумма 1 500,50 руб., т.е. 1 тыс. 500 руб. 50 коп., см. пп. 2.1.1 и ст. 450 ГК РФ [doc:abc:p1].",
                 "Договор подписали Иванов И. И. и Петров П. П. в г. Тверь 01.03.2025 г. [doc:abc:p1].",
                 "1. Ставка 700 руб. [doc:abc:p1]\n2. Срок до 31.12.2026 [doc:abc:p1]"]:
        block = f"<!-- llm -->{text}<!-- /llm -->"
        assert verify_claims(block, {"doc:abc:p1"}) == (block, 0), text


def test_sentence_ends_after_an_abbreviation_before_a_capital():
    md = "<!-- llm -->Ставка 900 руб. Срок до 2027 г. Договор действует [doc:abc:p1].<!-- /llm -->"
    assert verify_claims(md, {"doc:abc:p1"}) == ("<!-- llm -->Договор действует [doc:abc:p1].<!-- /llm -->", 2)


def test_text_outside_llm_blocks_untouched():
    out, removed = verify_claims("| Ставка | 900 |\n<!-- llm -->Без чисел.<!-- /llm -->", set())
    assert removed == 0 and "| Ставка | 900 |" in out


def test_snapshot():
    assert render_report(small_context()) == (SNAP / "report_small.md").read_text(encoding="utf-8")
