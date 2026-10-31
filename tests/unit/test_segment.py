from contract_analyst.ingest.segment import segment_clauses
from contract_analyst.models import Page, PageQuality


def _page(text, number=1):
    return Page(number=number, method="text", text=text, quality=PageQuality(chars=len(text)))


def test_segment_clauses_numbered_and_appendix():
    text = ("Преамбула\n1. Предмет\n1.1. Текст\n1.25. Группа тарифов\n12.2. Уведомления\n"
            "Приложение № 7\nТаблица")
    assert [c.clause_id for c in segment_clauses([_page(text)])] == [
        "preamble", "1", "1.1", "1.25", "12.2", "Приложение 7"]


def test_dates_and_day_numbers_do_not_start_clauses():
    clauses = segment_clauses([_page("1. Срок\n15 декабря 2015 года\n19.08.2020 г.\n109147, г. Тверь")])
    assert [c.clause_id for c in clauses] == ["1"]


def test_clause_keeps_start_page_and_spans_pages():
    clauses = segment_clauses([_page("1. Начало пункта", 1), _page("продолжение\n2. Следующий", 2)])
    assert [(c.clause_id, c.page) for c in clauses] == [("1", 1), ("2", 2)]
    assert "продолжение" in clauses[0].text
