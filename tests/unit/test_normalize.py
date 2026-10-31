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
