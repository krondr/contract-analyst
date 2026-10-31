import shutil
import unicodedata

from contract_analyst.ingest.inventory import list_inputs
from contract_analyst.models import DocFormat
from tests.fixtures.builders import make_text_pdf


def test_inventory_skips_system_files_marks_unsupported_and_duplicates(tmp_path):
    (tmp_path / "sub").mkdir()
    make_text_pdf(tmp_path / "a.pdf", ["Договор № D990000001-01"])
    shutil.copy(tmp_path / "a.pdf", tmp_path / "sub" / "a (2) (1).pdf")
    (tmp_path / ".DS_Store").write_bytes(b"x")
    (tmp_path / "~$draft.docx").write_bytes(b"x")
    (tmp_path / "t.xlsx").write_bytes(b"x")
    files = {f.name: f for f in list_inputs(tmp_path)}
    assert set(files) == {"a.pdf", "a (2) (1).pdf", "t.xlsx"}
    assert files["t.xlsx"].format == DocFormat.UNSUPPORTED
    assert files["a (2) (1).pdf"].duplicate_of == files["a.pdf"].path
    assert files["a.pdf"].duplicate_of is None and len(files["a.pdf"].sha256) == 64


def test_nfd_name_is_reported_in_nfc(tmp_path):
    nfd = unicodedata.normalize("NFD", "Договор й+ДС.pdf")
    make_text_pdf(tmp_path / nfd, ["Договор № D990000001-01"])
    (f,) = list_inputs(tmp_path)
    assert f.name == unicodedata.normalize("NFC", "Договор й+ДС.pdf")
