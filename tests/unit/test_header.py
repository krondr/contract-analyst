from datetime import date

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
