"""Individual grounding and plausibility checks. Each returns ok + a machine-readable code."""

from __future__ import annotations

import re
from datetime import date
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel
from rapidfuzz import fuzz

from contract_analyst.ingest.normalize import is_redacted, normalize_text
from contract_analyst.models import OpType, Page, PageMethod
from contract_analyst.parsing.patterns import (
    DOC_NUMBER_RE,
    date_spans,
    doc_number_spans,
    ends_table_row,
    find_dates,
    find_doc_numbers,
    find_numbers,
    number_readings,
)

ValueKind = Literal["date", "number", "money", "doc_number", "text"]
Span = tuple[int, int]  # offsets in normalize_text(page.text)
Figure = date | Decimal | str  # what a value states and the page must show: a date, an amount, a document number

MIN_DATE, MAX_DATE = date(2000, 1, 1), date(2035, 12, 31)
_TYPOGRAPHY = str.maketrans({"«": '"', "»": '"', "“": '"', "”": '"', "„": '"', "–": "-", "—": "-"})
_TOKEN_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё-]{3,}")
_ISO_DATE = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")
# A sentence ends at ; or at . ! ? after a word of four letters or more (so «п.», «г.», «руб.» do not end one),
# and before a line that starts a numbered clause.
_SENTENCE_END = re.compile(r"(?<=[^\W\d_]{4})[.!?](?=\s|$)|;|\n(?=\d{1,3}(?:\.\d{1,3})*\.?\s)")
_IN_FORCE = re.compile(r"вступа\w*\s+в\s+(?:силу|действие)|распростран\w*", re.IGNORECASE)
_REFERENCE = re.compile(r"\bот\s*$", re.IGNORECASE)  # «Договор от 21.01.2021» names a document, not a period
# Wording that carries an operation out, per operation type: every pattern must match. «Договор может быть
# расторгнут» only allows a termination, so permissive or conditional wording in the sentence refuses it.
_AGREED_ON = r"(?:договорились|пришли\s+к\s+соглашению|достигли\s+соглашения)\s+об?\s+(?:[а-яё]+\s+){0,2}?"
_STOPS = r"прекрат(?:ить|ил\w*)|прекраща(?:ет|ют|ется|ются)\b"
_OPERATIVE = {
    OpType.TERMINATE: [rf"расторг(?:нуть|ается|аются|нут\w*)|{_STOPS}|{_AGREED_ON}(?:расторжени\w*|прекращени\w*)"],
    OpType.SUSPEND_PART: [rf"{_STOPS}|приостан(?:овить|овлен\w*|авлива\w*)|{_AGREED_ON}(?:прекращени\w*|приостановлени\w*)"],
    OpType.RESTATE_DOCUMENT: [r"изложить|излага(?:ется|ются)|изложен\w*|читать|принять", r"редакци"],
    # A clause operation is stated by its own verb: a sentence that merely names a clause does not change it.
    OpType.REPLACE_CLAUSE: [r"излож\w*|излага\w*|читать|замени\w*|редакци"],
    OpType.RESTATE_SECTION: [r"излож\w*|излага\w*|читать|замени\w*|редакци"],
    OpType.REPLACE_APPENDIX: [r"излож\w*|излага\w*|читать|замени\w*|редакци"],
    OpType.ADD_CLAUSE: [r"дополни\w*|дополня\w*|добави\w*|включи\w*"],
    OpType.DELETE_CLAUSE: [r"исключи\w*|исключа\w*|утративш\w*\s+силу|утрачива\w*\s+силу|удали\w*"],
}
_PERMISSIVE = re.compile(r"\bмо(?:жет|гут)\b|\bвправе\b|\bиме(?:ет|ют)\s+право\b|\bв\s+случа(?:е|ях)\b"
                         r"|\bпри\s+условии\b|\bесли\b|\bподлеж\w*", re.IGNORECASE)
_NEGATED = re.compile(r"\bне\s+(?:\S+\s+)?$", re.IGNORECASE)


class CheckResult(BaseModel):
    ok: bool
    code: str
    reason: str | None = None


def _norm(text: str) -> str:
    return " ".join(normalize_text(text).translate(_TYPOGRAPHY).casefold().split())


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(_norm(text))


def _ok(code: str) -> CheckResult:
    return CheckResult(ok=True, code=code)


def _fail(code: str, reason: str) -> CheckResult:
    return CheckResult(ok=False, code=code, reason=reason)


@lru_cache(maxsize=256)
def _indexed(text: str) -> tuple[str, str, tuple[int, ...]]:
    """normalize_text(text); _norm(text); for every character of the latter, its offset in the former."""
    source = normalize_text(text)
    chars: list[str] = []
    index: list[int] = []
    for i, ch in enumerate(source):
        if ch.isspace():
            if chars and chars[-1] != " ":
                chars.append(" ")
                index.append(i)
            continue
        for folded in ch.translate(_TYPOGRAPHY).casefold():
            chars.append(folded)
            index.append(i)
    if chars and chars[-1] == " ":
        chars.pop()
        index.pop()
    return source, "".join(chars), tuple(index)


def _whole_words(text: str, start: int, end: int) -> Span:
    """A quote that starts inside "21500" is no evidence for 1500: spans grow to whole words."""
    while start > 0 and not text[start - 1].isspace():
        start -= 1
    while end < len(text) and not text[end].isspace():
        end += 1
    return start, end


def locate_quote(quote: str | None, page: Page, fuzzy_threshold: int) -> tuple[CheckResult, list[Span]]:
    """Where the quote is on the page: every verbatim occurrence, or on a scan the fuzzily aligned part."""
    if not quote or not quote.strip():
        return _fail("no_quote", "цитата не указана"), []
    q = _norm(quote)
    source, flat, index = _indexed(page.text)
    spans: list[Span] = []
    at = flat.find(q)
    while at >= 0:
        spans.append(_whole_words(source, index[at], index[at + len(q) - 1] + 1))
        at = flat.find(q, at + 1)
    if spans:
        return _ok("quote_present"), spans
    if page.method == PageMethod.OCR:
        aligned = fuzz.partial_ratio_alignment(q, flat, score_cutoff=fuzzy_threshold)
        if aligned is not None and aligned.dest_end > aligned.dest_start:
            span = _whole_words(source, index[aligned.dest_start], index[aligned.dest_end - 1] + 1)
            return _ok("quote_present_fuzzy"), [span]
    return _fail("quote_not_found", f"цитата не найдена на странице {page.number}"), []


def check_quote_present(quote: str | None, page: Page, fuzzy_threshold: int) -> CheckResult:
    return locate_quote(quote, page, fuzzy_threshold)[0]


def _as_date(value: str) -> date | None:
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _as_decimal(value: str) -> Decimal | None:
    try:
        return Decimal(value.replace(" ", "").replace(",", "."))
    except InvalidOperation:
        return None


def _figures(text: str, split_cells: bool = True) -> list[tuple[Figure, int, int, bool]]:
    """Dates, document numbers and amounts with positions and ambiguity; the digits of a date or a document
    number are part of it, not amounts."""
    found: list[tuple[Figure, int, int, bool]] = [(d, a, b, False) for d, a, b in date_spans(text)]
    found += [(n, a, b, False) for n, a, b in doc_number_spans(text)]
    taken = [(a, b) for _, a, b, _ in found]
    found += [(r.value, r.start, r.end, r.ambiguous) for r in number_readings(text, split_cells)
              if not any(a < r.end and r.start < b for a, b in taken)]
    return sorted(found, key=lambda f: f[1])


def _value_figures(value: str, kind: ValueKind) -> list[Figure]:
    if kind == "date":
        return [d] if (d := _as_date(value)) else []
    if kind in ("money", "number"):
        return [n] if (n := _as_decimal(value)) is not None else []
    if kind == "doc_number":
        return [value]
    return [f for f, *_ in _figures(_as_written(value), split_cells=False)]


def _as_written(value: str) -> str:
    """A text value with the model's ISO dates written the way contracts write them."""
    return _ISO_DATE.sub(lambda m: f"{m[3]}.{m[2]}.{m[1]}", normalize_text(value))


def _show(figure: Figure) -> str:
    return figure.strftime("%d.%m.%Y") if isinstance(figure, date) else str(figure)


def check_value_in_quote(value: str, quote: str | None, kind: ValueKind) -> CheckResult:
    if not quote:
        return _fail("no_quote", "цитата не указана")
    if kind == "date":
        d = _as_date(value)
        found = [x for x, _ in find_dates(normalize_text(quote))]
        return _ok("value_in_quote") if d and d in found else _fail("value_not_in_quote", f"даты {value} нет в цитате")
    if kind in ("money", "number"):
        n = _as_decimal(value)
        found = find_numbers(normalize_text(quote))
        return _ok("value_in_quote") if n is not None and n in found else _fail(
            "value_not_in_quote", f"числа {value} нет в цитате")
    if kind == "doc_number":
        return _ok("value_in_quote") if value in find_doc_numbers(normalize_text(quote)) else _fail(
            "value_not_in_quote", f"номера {value} нет в цитате")
    quoted_figures = {f for f, *_ in _figures(normalize_text(quote))}
    if missing := [f for f in _value_figures(value, "text") if f not in quoted_figures]:
        return _fail("value_not_in_quote", f"{_show(missing[0])} из значения нет в цитате")
    tokens = _tokens(_as_written(value))
    if not tokens:
        return _ok("value_in_quote")
    quoted = _norm(quote)
    share = sum(1 for t in tokens if t in quoted) / len(tokens)
    return _ok("value_in_quote") if share >= 0.6 else _fail("value_not_in_quote", "значение не следует из цитаты")


def _word_confidences(page: Page) -> list[tuple[int, int, float | None]]:
    """Every word of normalize_text(page.text) with the confidence of the OCR word at the same position."""
    source = normalize_text(page.text)
    words = [(m.start(), m.end()) for m in re.finditer(r"\S+", source)]
    confs: list[float | None] = [None] * len(words)
    matcher = SequenceMatcher(None, [_norm(source[a:b]) for a, b in words], [_norm(w.text) for w in page.words],
                              autojunk=False)
    for block in matcher.get_matching_blocks():
        for k in range(block.size):
            confs[block.a + k] = page.words[block.b + k].conf
    return [(a, b, conf) for (a, b), conf in zip(words, confs)]


def confident(start: int, end: int, words: list[tuple[int, int, float | None]], threshold: float) -> bool:
    """Every OCR word the text at start:end is read from is recognized at or above the threshold."""
    return all(conf is not None and conf >= threshold for a, b, conf in words if a < end and start < b)


def check_value_on_page(value: str, kind: ValueKind, page: Page, spans: list[Span],
                        conf_threshold: float) -> CheckResult:
    """The value is read from the page, not from the model's quote: every figure it states must occur in a
    located quote span, on a scan in confidently recognized words, and in a reading that is not ambiguous."""
    source = normalize_text(page.text)
    on_page = [f for f in _figures(source) if any(a <= f[1] and f[2] <= b for a, b in spans)]
    words = _word_confidences(page) if page.method == PageMethod.OCR else None
    ambiguous: Figure | None = None
    for figure in _value_figures(value, kind):
        seen = [(a, b, amb) for f, a, b, amb in on_page if f == figure]
        if not seen:
            return _fail("value_not_on_page", f"{_show(figure)} нет на странице {page.number} в цитируемом месте")
        if words is not None:
            seen = [(a, b, amb) for a, b, amb in seen if confident(a, b, words, conf_threshold)]
            if not seen:
                return _fail("ocr_value_unconfirmed", f"OCR не уверен в «{_show(figure)}» на странице {page.number}")
        if all(amb for *_, amb in seen):
            ambiguous = figure
    if ambiguous is not None:
        return _fail("ambiguous_number", f"«{_show(ambiguous)}» может быть частью числа или соседней ячейкой таблицы")
    return _ok("value_on_page")


def sentence_around(text: str, start: int, end: int) -> Span:
    """The sentence or sentences that hold text[start:end]."""
    begin = 0
    for m in _SENTENCE_END.finditer(text, 0, start):
        begin = m.end()
    m = _SENTENCE_END.search(text, end)
    return begin, m.end() if m else len(text)


def _lines(text: str) -> list[Span]:
    spans, start = [], 0
    for line in text.split("\n"):
        spans.append((start, start + len(line)))
        start += len(line) + 1
    return spans


def _dates_between(text: str, start: int, end: int, words: list | None, threshold: float,
                   references: bool = True) -> set[date]:
    """Dates stated inside text[start:end]; on a scan only confidently recognized ones."""
    return {d for d, a, b in date_spans(text) if start <= a and b <= end
            and (references or not _REFERENCE.search(text, max(0, a - 6), a))
            and (words is None or confident(a, b, words, threshold))}


def _header_dates(text: str, lines: list[Span], row: int, words: list | None, threshold: float) -> set[date]:
    """Dates of the nearest dated line above a table row, with the dated lines right above it: the header of the
    row's columns or section."""
    found: set[date] = set()
    for i in range(row - 1, -1, -1):
        if dates := _dates_between(text, *lines[i], words, threshold):
            found |= dates
        elif found:
            break
    return found


def check_period_near(amount: Decimal, period: list[date | None], page: Page, spans: list[Span],
                      conf_threshold: float) -> CheckResult:
    """A rate's period dates must be stated next to the amount: on its row, in the nearest dated header above
    it, or — for a rate in prose — in its sentence. Somewhere else in the document is not enough."""
    wanted = [d for d in period if d]
    if not wanted:
        return _ok("period_near")
    source = normalize_text(page.text)
    words = _word_confidences(page) if page.method == PageMethod.OCR else None
    lines = _lines(source)
    places = [(a, b) for f, a, b, amb in _figures(source)
              if f == amount and not amb and any(s <= a and b <= e for s, e in spans)]
    for a, b in places:
        row = next(i for i, (s, e) in enumerate(lines) if s <= a <= e)
        near = _dates_between(source, *lines[row], words, conf_threshold)
        near |= _header_dates(source, lines, row, words, conf_threshold)
        if not ends_table_row(source, b):
            near |= _dates_between(source, *sentence_around(source, a, b), words, conf_threshold)
        if all(d in near for d in wanted):
            return _ok("period_near")
    shown = " – ".join(_show(d) for d in wanted)
    return _fail("date_not_near", f"период {shown} не указан рядом со ставкой {amount} на странице {page.number}")


def check_op_dates_near(dates: list[date | None], page: Page, spans: list[Span], pages: list[Page],
                        conf_threshold: float) -> CheckResult:
    """An operation's dates must be stated in its own sentence. Only when that sentence names no date (other than
    references such as «Договор от 21.01.2021») may the document's entry-into-force statement give them."""
    wanted = [d for d in dates if d]
    if not wanted:
        return _ok("dates_near")
    source = normalize_text(page.text)
    words = _word_confidences(page) if page.method == PageMethod.OCR else None
    stated: set[date] = set()
    for a, b in spans:
        stated |= _dates_between(source, *sentence_around(source, a, b), words, conf_threshold, references=False)
    if not stated:
        for other in pages:
            text = normalize_text(other.text)
            other_words = _word_confidences(other) if other.method == PageMethod.OCR else None
            for m in _IN_FORCE.finditer(text):
                stated |= _dates_between(text, *sentence_around(text, m.start(), m.end()), other_words, conf_threshold,
                                         references=False)
    if missing := [d for d in wanted if d not in stated]:
        return _fail("date_not_near", f"даты {_show(missing[0])} нет в предложении с операцией на странице {page.number}"
                                      " и в условии о вступлении в силу")
    return _ok("dates_near")


def _says(pattern: str, text: str) -> bool:
    return any(not _NEGATED.search(text, 0, m.start()) for m in re.finditer(pattern, text, re.IGNORECASE))


def check_operative_wording(op: OpType, page: Page, spans: list[Span]) -> CheckResult:
    """The sentence quoted for an operation must carry it out: its own verb, not a possibility or a condition.
    Termination, suspension and restatement change everything after them; a clause operation changes the fold."""
    patterns = _OPERATIVE.get(op)
    if not patterns:
        return _ok("operative")
    source = normalize_text(page.text)
    for a, b in spans:
        sentence = source[slice(*sentence_around(source, a, b))]
        if all(_says(p, sentence) for p in patterns) and not _PERMISSIVE.search(sentence):
            return _ok("operative")
    return _fail("not_operative", f"в цитируемом предложении нет действующей формулировки для «{op.value}»: "
                                  "только возможность или условие")


def check_format(value: str, kind: ValueKind) -> CheckResult:
    if kind == "doc_number":
        return _ok("format") if DOC_NUMBER_RE.fullmatch(value) else _fail("bad_format", f"номер {value} не по формату")
    if kind == "date":
        d = _as_date(value)
        return _ok("format") if d and MIN_DATE <= d <= MAX_DATE else _fail("bad_format", f"дата {value} вне диапазона")
    if kind in ("money", "number"):
        n = _as_decimal(value)
        if n is None or (kind == "money" and n < 0):
            return _fail("bad_format", f"сумма {value} некорректна")
        return _ok("format")
    return _ok("format") if value.strip() else _fail("bad_format", "пустое значение")


def check_ocr_confidence(quote: str, page: Page, threshold: float) -> CheckResult:
    if page.method != PageMethod.OCR:
        return _ok("not_ocr")
    by_text: dict[str, float] = {}
    for word in page.words:
        key = _norm(word.text)
        by_text[key] = max(by_text.get(key, 0.0), word.conf)
    confs = [by_text[t] for t in _norm(quote).split() if t in by_text]
    if not confs:
        return _fail("ocr_low_confidence", "уверенность OCR для цитаты не подтверждена")
    low = min(confs)
    return _ok("ocr_confident") if low >= threshold else _fail(
        "ocr_low_confidence", f"уверенность OCR {low:.2f} ниже порога {threshold:.2f}")


def check_not_redacted(quote: str | None, value: str | None) -> CheckResult:
    if (quote and is_redacted(quote)) or (value and is_redacted(value)):
        return _fail("redacted", "поле вымарано в источнике")
    return _ok("not_redacted")


# The role a date plays is read from the words right before it: «вступает в силу с», «до», «к Договору № … от».
_ROLE_CUT = re.compile(r"[«»\"“”„]")
_REFERENCE_BEFORE = re.compile(r"\b(?:к|по|в|о\s+расторжении)\s+(?:договор\w*|соглашени\w*|контракт\w*)\b[^.;]{0,70}?\bот\s*$")
_START_BEFORE = re.compile(r"(?:\bс|\bсо|вступ\w*\s+в\s+силу(?:\s+с)?)\s*[:—-]?\s*$")
_START_CONTEXT = re.compile(r"вступ\w*\s+в\s+силу|распространя\w*|применя\w*\s+с\b|действ\w*\s+с\b")
_END_BEFORE = re.compile(r"(?:\bдо|\bпо|окончани\w*|истечени\w*|срок\w*\s+действи\w*)\s*[:—-]?\s*$")
DATE_ROLES = {"doc_date", "effective_from", "effective_to"}
_OPERATION_VERB = re.compile(r"расторг\w*|прекра[тщ]\w*|приостан\w*|излож\w*|исключ\w*|дополн\w*|отмен\w*")
_IN_FORCE_CUE = re.compile(r"вступ\w*\s+в\s+силу|распространя\w*")


def check_date_role(field: str, value: str | None, quote: str | None) -> CheckResult:
    """A date found in the quote must be there in the field's role: a signing date is not the date of the contract a
    supplement refers to, an entry into force is a start (not an end, not a reference), an end date is an end."""
    target = _as_date(value or "")
    if field not in DATE_ROLES or target is None or not quote:
        return _ok("date_role_unchecked")
    windows = [" ".join(_ROLE_CUT.sub(" ", quote[max(0, start - 90):start]).casefold().split()) + " "
               for d, start, _ in date_spans(quote) if d == target]
    if not windows:
        return _ok("date_role_unchecked")  # value_in_quote decides whether the date is there at all
    if field == "doc_date":
        fits = [w for w in windows if not (_REFERENCE_BEFORE.search(w) or _START_BEFORE.search(w) or _END_BEFORE.search(w))]
        reason = "дата в цитате — не дата подписания этого документа (ссылка на другой договор или срок действия)"
    elif field == "effective_from":
        folded = " ".join(quote.casefold().split())
        # «расторгают Договор с 01.09.2026» dates the termination, not the supplement's own entry into force.
        if _OPERATION_VERB.search(folded) and not _IN_FORCE_CUE.search(folded):
            return _fail("date_role", "дата в цитате — дата операции (расторжение, изменение), а не вступления в силу")
        fits = [w for w in windows if not (_REFERENCE_BEFORE.search(w) or _END_BEFORE.search(w))
                and (_START_BEFORE.search(w) or _START_CONTEXT.search(w))]
        reason = "дата в цитате — не начало действия (окончание срока, ссылка или нет слов «вступает в силу», «с»)"
    else:
        fits = [w for w in windows if _END_BEFORE.search(w)]
        reason = "дата в цитате — не окончание действия (нет слов «до», «по», «срок действия»)"
    return _ok("date_role") if fits else _fail("date_role", reason)
