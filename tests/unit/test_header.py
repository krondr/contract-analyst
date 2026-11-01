from datetime import date

import pytest

from contract_analyst.parsing.header import detect_draft, parse_header


def test_header_supplement():
    h = parse_header("Дополнительное соглашение № D990000002-01\n"
                     "к Договору коммерческого представительства № D990000001-01 от «21» января 2021 г.\n"
                     "г. Тверь «07» июля 2021 г.\n")
    assert (h.title_kind, h.number) == ("supplement", "D990000002-01")
    assert (h.parent_number, h.parent_date) == ("D990000001-01", date(2021, 1, 21))
    assert (h.city, h.doc_date) == ("Тверь", date(2021, 7, 7))


def test_header_number_glued_to_sign():
    h = parse_header("Договор №D990000003-01\nг. Тверь 19.08.2020 г.")
    assert (h.title_kind, h.number, h.doc_date) == ("contract", "D990000003-01", date(2020, 8, 19))


def test_header_amendment_wording():
    h = parse_header("ДОПОЛНИТЕЛЬНОЕ СОГЛАШЕНИЕ № D990000005-01\nо внесении изменений в договор\n"
                     "от 23.08.2022 № D990000004-01\nг. Тверь")
    assert (h.parent_number, h.parent_date) == ("D990000004-01", date(2022, 8, 23))


def test_header_agreement_with_latin_c():
    h = parse_header("Cоглашение № D990000006-01\nг. Псков «13» августа 2021 г.")
    assert (h.title_kind, h.number, h.city) == ("agreement", "D990000006-01", "Псков")
    assert h.spans["number"].startswith("Соглашение")


def test_detect_draft():
    assert detect_draft("Проект Договора.pdf", "Договор № D990000003-01")
    assert detect_draft("x.pdf", "ПРОЕКТ\nДоговор № D990000003-01")
    assert not detect_draft("D990000003-01.pdf", "Договор № D990000003-01\nпроект торговой точки")


def test_header_on_ocr_text_with_misread_d():
    h = parse_header("ДОГОВОР № 0990000018-01\nк Договору № 0980000016-09 от 15.12.2015 г.")
    assert (h.number, h.parent_number) == ("D990000018-01", "D980000016-09")


def test_header_title_with_adjective_and_termination_parent():
    h = parse_header("ДИЛЕРСКИЙ ДОГОВОР № D990300001-01\nг. Тверь «12» мая 2022 г.")
    assert (h.title_kind, h.number) == ("contract", "D990300001-01")
    s = parse_header("Соглашение № D990200002-01\nо расторжении Договора оказания услуг № D990200001-01 от 15.02.2023")
    assert (s.parent_number, s.parent_date) == ("D990200001-01", date(2023, 2, 15))


@pytest.mark.parametrize("adjective", ["Агентскому", "Дилерскому", "Лицензионному", "Партнёрскому"])
def test_header_parent_contract_named_with_an_adjective(adjective):
    h = parse_header(f"ДОПОЛНИТЕЛЬНОЕ СОГЛАШЕНИЕ № D990000002-01\nк {adjective} договору № D990000001-01 от 10.01.2024 г.\n"
                     "г. Тверь 15.10.2026 г.")
    assert (h.parent_number, h.parent_date) == ("D990000001-01", date(2024, 1, 10))
