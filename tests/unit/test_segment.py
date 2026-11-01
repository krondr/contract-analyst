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


def test_clauses_after_an_appendix_heading_belong_to_it():
    text = ("12.2. Уведомления\nПриложение № 1\nк Договору № D990000001-01\n1.6. Вознаграждение\n"
            "Приложение № 3 к Договору\n2.1. Торговые точки")
    assert [(c.clause_id, c.appendix) for c in segment_clauses([_page(text)])] == [
        ("12.2", None), ("Приложение 1", "Приложение 1"), ("1.6", "Приложение 1"),
        ("Приложение 3", "Приложение 3"), ("2.1", "Приложение 3")]


def test_appendix_named_in_a_sentence_opens_no_appendix():
    clauses = segment_clauses([_page("5.1. Оплата\nПриложение № 1 является неотъемлемой частью Договора.\n5.2. Срок")])
    assert [c.appendix for c in clauses if c.clause_id == "5.2"] == [None]


def test_clause_at_names_the_clause_that_holds_a_quote():
    from contract_analyst.gate.gate import clause_at
    from tests.fixtures.builders import text_doc

    doc = text_doc([("12.1. Уведомления направляются по факсу.\n12.2. Копии имеют силу оригинала.\n"
                     "Приложение № 1\nк Договору\n1.6. Вознаграждение составляет 60% от начислений.")])
    assert clause_at(doc, 1, "Копии имеют силу оригинала") == "12.2"
    assert clause_at(doc, 1, "Вознаграждение составляет 60%") == "Приложение 1/1.6"
    assert clause_at(doc, 1, "этого текста в документе нет") is None
