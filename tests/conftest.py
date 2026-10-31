"""Shared fixtures: a small fictional document package (contract, supplement, scanned supplement)."""

from pathlib import Path

import pytest

from tests.fixtures.builders import make_scan_pdf, make_text_pdf

MAIN_TEXT = ("Договор аренды № D990000001-01\nг. Тверь 10.01.2024 г.\n"
             "1. Предмет договора: аренда нежилого помещения.\n"
             "3.1. Арендная плата составляет 100000 рублей в месяц без НДС.\n"
             "7.1. Договор действует до 31.12.2025.")
SUPPLEMENT_TEXT = ("Дополнительное соглашение № D990000002-01\n"
                   "к Договору аренды № D990000001-01 от 10.01.2024 г.\ng. Тверь 01.03.2025 г.\n"
                   "1. Изложить п. 3.1 Договора в следующей редакции: «3.1. Арендная плата составляет "
                   "120000 рублей в месяц без НДС».\n2. Настоящее Соглашение вступает в силу с 01.04.2025.")
SCAN_TEXT = ("Дополнительное соглашение № D990000003-01\nк Договору аренды № D990000001-01 от 10.01.2024 г.\n"
             "1. Изложить п. 7.1 Договора: срок действия до 31.12.2026.")


@pytest.fixture(scope="session")
def fixtures_package(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("package")
    make_text_pdf(root / "Договор аренды.pdf", [MAIN_TEXT])
    make_text_pdf(root / "ДС1.pdf", [SUPPLEMENT_TEXT])
    make_scan_pdf(root / "ДС2 скан.pdf", [SCAN_TEXT])
    return root
