from datetime import date
from decimal import Decimal

import pytest

from contract_analyst.parsing.patterns import (
    date_spans,
    doc_number_spans,
    find_dates,
    find_doc_numbers,
    find_numbers,
    number_readings,
    parse_money,
    parse_ru_date,
)


@pytest.mark.parametrize("s,expected", [
    ("«21» января 2021 г.", date(2021, 1, 21)), ("19.08.2020", date(2020, 8, 19)),
    ("23.08.2022г.", date(2022, 8, 23)), ("6.02.2024г.", date(2024, 2, 6)),
    ("с 01 июля 2026 года", date(2026, 7, 1)), ("«07» июля 2021 г.", date(2021, 7, 7))])
def test_parse_ru_date(s, expected):
    assert parse_ru_date(s) == expected


def test_parse_ru_date_rejects_impossible_date():
    assert parse_ru_date("31.02.2024") is None


def test_find_dates_returns_spans_in_order():
    found = find_dates("с 01.07.2026 по 31 декабря 2026 года")
    assert [d for d, _ in found] == [date(2026, 7, 1), date(2026, 12, 31)]


def test_find_doc_numbers_handles_glued_sign():
    assert find_doc_numbers("Договор №D990000003-01 и ДС № D990000004") == ["D990000003-01", "D990000004"]


def test_parse_money_ru_format():
    assert parse_money("330,5") == Decimal("330.5")
    assert parse_money("1 500") == Decimal(1500)
    assert parse_money("3000 (три тысячи) рублей") == Decimal(3000)
    assert parse_money("нет суммы") is None


def test_find_numbers_keeps_both_readings_of_space_separated_table_cells():
    found = find_numbers("Тариф Старт 270 500")
    assert Decimal(270) in found and Decimal(500) in found and Decimal(270500) in found
    assert find_numbers("Пакет Мини 675 1000") == [Decimal(675), Decimal(1000)]
    assert parse_money("1 500 рублей") == Decimal(1500)


@pytest.mark.parametrize("prose", [
    "Вознаграждение составляет 1 500 рублей за абонента.", "в размере 1 500\nрублей за абонента",
    "1 500 (одна тысяча пятьсот) рублей", "Ставка составляет 1 500.\nНастоящее соглашение"])
def test_grouped_number_in_prose_is_one_amount(prose):
    assert find_numbers(prose) == [Decimal(1500)]


def test_table_row_cells_on_one_line_are_ambiguous_readings():
    text = "Тариф Старт 270 500\nОптима 330,5 500"
    readings = number_readings(text)
    assert [(r.value, r.ambiguous) for r in readings] == [
        (Decimal(270500), True), (Decimal(270), True), (Decimal(500), True),
        (Decimal("330.5"), False), (Decimal(500), False)]
    assert [text[r.start:r.end] for r in readings] == ["270 500", "270", "500", "330,5", "500"]


def test_explicit_cell_separator_keeps_grouped_numbers_whole():
    assert find_numbers("Безлимит | 1 200 | 1 500") == [Decimal(1200), Decimal(1500)]
    assert not any(r.ambiguous for r in number_readings("Безлимит | 1 200 | 1 500"))


def test_spans_point_into_the_text():
    text = "Соглашение № D990000003-01 от «21» января 2021 г. и от 01.02.2021, ставка 700 рублей"
    assert [(n, text[a:b]) for n, a, b in doc_number_spans(text)] == [("D990000003-01", "D990000003-01")]
    assert [(d, text[a:b]) for d, a, b in date_spans(text)] == [
        (date(2021, 1, 21), "«21» января 2021"), (date(2021, 2, 1), "01.02.2021")]


def test_ocr_misread_leading_d_is_normalized():
    assert find_doc_numbers("Соглашение № 0990000017-09 к Договору №0980000016-09") == ["D990000017-09", "D980000016-09"]
    assert find_doc_numbers("ДОГОВОР № Д990000018-01") == ["D990000018-01"]
    assert find_doc_numbers("индекс 109147 и счёт 0123456789") == []
