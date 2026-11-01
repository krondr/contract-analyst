from collections import Counter

import numpy as np

from contract_analyst.eval.ocr_bench import (
    BenchRow,
    SyntheticScan,
    bench,
    crit_curve,
    critical_tokens,
    critical_words,
    make_synthetic_scans,
    render_markdown,
    run_metadata,
    select_winner,
    threshold_at_precision,
)
from contract_analyst.models import OcrWord
from contract_analyst.ocr.base import OcrResult
from tests.fixtures.builders import make_text_pdf

PAGES_A = [
    "Дополнительное соглашение № D990000010-01\nг. Тверь 19.08.2020 г.\nЦена товара составляет 1500 рублей.",
    "2. Срок поставки — 30 дней с даты заявки.\n3. Неустойка 0,1% за каждый день просрочки.",
]
PAGES_B = [
    "Договор поставки № D990000001-01\nг. Псков 01.02.2021 г.\nПокупатель оплачивает 2 400,50 рублей.",
    "4. Договор действует до 31.12.2027 г.\n5. Споры рассматриваются в арбитражном суде.",
]


def test_select_winner_rule_order():
    rows = [BenchRow("tesseract", "none", .05, .1, .90, .80, 1.0), BenchRow("rapidocr", "none", .03, .08, .90, .85, 2.0),
            BenchRow("rapidocr", "deskew_sauvola", .04, .09, .95, .82, 3.0)]
    winner = select_winner(rows)
    assert (winner.engine, winner.mode) == ("rapidocr", "deskew_sauvola")  # critical coverage beats CER

    # coverage 0.95 vs 0.94 is still a tie (difference <= 0.01): the lower CER wins
    rows = [BenchRow("tesseract", "none", .05, .1, .95, .80, 1.0), BenchRow("rapidocr", "none", .03, .08, .94, .85, 2.0)]
    assert select_winner(rows).engine == "rapidocr"

    # coverage and CER tied: the faster variant wins
    rows = [BenchRow("tesseract", "none", .03, .1, .95, .80, 2.0), BenchRow("rapidocr", "none", .03, .08, .95, .85, 1.0)]
    assert select_winner(rows).engine == "rapidocr"


def test_synthetic_scans_deterministic(tmp_path):
    pdfs = [make_text_pdf(tmp_path / "a.pdf", PAGES_A), make_text_pdf(tmp_path / "b.pdf", PAGES_B)]
    first = make_synthetic_scans(pdfs, n_pages=4, seed=7)
    again = make_synthetic_scans(list(reversed(pdfs)), n_pages=4, seed=7)  # input order does not matter

    assert len(first) == len(again) == 4
    for a, b in zip(first, again, strict=True):
        assert np.array_equal(a.image, b.image) and a.truth == b.truth and a.meta == b.meta
    assert sum(s.meta["rot90"] != 0 for s in first) == 1  # 25% of the pages are turned by 90 degrees
    assert all(s.image.shape[1] > s.image.shape[0] for s in first if s.meta["rot90"])
    assert any("D990000010-01" in s.truth for s in first)  # truth is the normalized text layer

    other = make_synthetic_scans(pdfs, n_pages=4, seed=8)
    assert any(a.image.shape != b.image.shape or not np.array_equal(a.image, b.image)
               for a, b in zip(first, other, strict=True))


def test_critical_tokens_extraction():
    text = ("Дополнительное соглашение № D990000010-01 от 19.08.2020 г. к договору D990000001.\n"
            "Цена 1 500,50 руб., п. 3.1; срок «15» дней (1500) 10%")
    assert critical_tokens(text) == ["D990000010-01", "19.08.2020", "D990000001", "1", "500,50", "3.1", "15", "1500",
                                     "10"]
    # whole words only: 10 digits after D, a Cyrillic "Д" or a glued unit make a word that is not a value
    assert critical_tokens("D9900000011 Д990000010-01 0990000010-01 1500руб.") == []

    words = [OcrWord(text="№D990000010-01,", conf=.9), OcrWord(text="1500руб.", conf=.8), OcrWord(text="(1500)", conf=.7),
             OcrWord(text="Д990000010-01", conf=.95), OcrWord(text="цена", conf=.6)]
    assert critical_words(words) == [("D990000010-01", .9), ("1500", .7)]


def test_threshold_is_smallest_t_reaching_target_precision():
    truth = Counter({"100": 1, "200": 1, "D990000010-01": 1})
    ocr = [("100", .95), ("200", .90), ("D990000010-01", .97), ("300", .70)]  # "300" is a misread
    assert threshold_at_precision(crit_curve([(ocr, truth)])) == (0.71, 1.0)

    # a duplicate is matched against the truth only once
    point = crit_curve([([("100", .9), ("100", .9)], Counter({"100": 1}))], grid=(0.5,))[0]
    assert (point.precision, point.coverage) == (0.5, 1.0)

    # a confident misread that no threshold on the grid removes: no threshold, zero coverage
    assert threshold_at_precision(crit_curve([(ocr + [("999", .995)], truth)])) == (None, 0.0)


class _EchoEngine:
    """Fake engine that 'reads' a fixed text with a fixed confidence."""

    name = "echo"

    def __init__(self, text: str, conf: float):
        self.text, self.conf = text, conf

    def recognize(self, image: np.ndarray) -> OcrResult:
        words = [OcrWord(text=w, conf=self.conf) for w in self.text.split()]
        return OcrResult(text=self.text, words=words, mean_conf=self.conf)


TRUTH = "Договор № D990000001-01 от 01.02.2021, цена 1500 рублей"


def _blank(meta: dict | None = None) -> SyntheticScan:
    return SyntheticScan(np.full((40, 40, 3), 255, np.uint8), TRUTH, meta or {})


def test_bench_scores_each_engine_and_mode():
    scans = [_blank(), _blank(), _blank({"rot90": 90})]  # the echo engine never turns a page back
    real = [SyntheticScan(np.full((40, 40, 3), 255, np.uint8), TRUTH.replace("1500", "1600"), {})]
    (row,) = bench([_EchoEngine(TRUTH, .9)], ["none"], scans, real)
    assert (row.engine, row.mode, row.cer, row.wer) == ("echo", "none", 0.0, 0.0)
    assert (row.crit_coverage_at_p99, row.threshold, row.recommended_threshold) == (1.0, 0.5, 0.55)
    assert row.n_pages == 3 and row.n_crit_truth == 9 and row.n_doc_numbers == 3 and row.sec_per_page >= 0
    assert row.doc_number_recall == 1.0 and row.rotation_fixed == 0.0 and row.cer_rotated == 0.0
    assert row.wer_unordered == 0.0
    assert row.n_real_pages == 1 and row.real_cer > 0 and row.real_crit_coverage == 2 / 3


def test_bench_interleaves_variants_but_keeps_their_results_apart():
    exact, sloppy = _EchoEngine(TRUTH, .9), _EchoEngine(TRUTH.replace("1500", "1300"), .9)
    sloppy.name = "sloppy"
    rows = bench([exact, sloppy], ["none"], [_blank(), _blank(), _blank()], [])
    assert [r.engine for r in rows] == ["echo", "sloppy"]
    assert rows[0].cer == 0.0 and rows[1].cer > 0.0 and rows[1].crit_coverage_at_p99 == 0.0


def test_results_doc_has_metrics_only():
    rows = bench([_EchoEngine(TRUTH, .9)], ["none"], [_blank(), _blank({"rot90": 270})], [])
    winner = select_winner(rows)
    doc = render_markdown(rows, winner, run_metadata([_blank(), _blank({"rot90": 270})], 1, 0, 0, 1.0))
    assert "ocr_engine: echo" in doc and "ocr_word_conf_threshold: 0.55" in doc
    for secret in ("D990000001-01", "01.02.2021", "1500", "Договор", "рублей"):
        assert secret not in doc
