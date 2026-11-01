import pytest

from contract_analyst.ingest.normalize import is_redacted, normalize_text


def test_latin_homoglyph_fixed_only_inside_cyrillic_words():
    assert normalize_text("Cоглашение № D990000001-01") == "Соглашение № D990000001-01"


def test_latin_only_words_untouched():
    assert normalize_text("ABC Ooo D990000001-01") == "ABC Ooo D990000001-01"


def test_hyphenated_line_break_joined():
    assert normalize_text("пред-\nставительство") == "представительство"


def test_spaces_collapsed_newlines_kept():
    assert normalize_text("а   б\n\nв") == "а б\n\nв"


@pytest.mark.parametrize("frag", ["- - - -", "________", "«  »", '" "'])
def test_redaction_markers(frag):
    assert is_redacted(frag)


def test_plain_text_is_not_redacted():
    assert not is_redacted("ООО «Ромашка»")


@pytest.mark.parametrize("frag", ["вознаграждение — ХХХХ рублей", "ставка XXX руб.", "в размере ххх,хх рублей"])
def test_letter_masks_are_redaction(frag):
    assert is_redacted(frag)


@pytest.mark.parametrize("frag", ["Договор № D990000001-01", "пункт XII Договора", "ООО «Ромашка»"])
def test_roman_numerals_and_names_are_not_redaction(frag):
    assert not is_redacted(frag)
