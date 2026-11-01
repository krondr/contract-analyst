"""Score a run against the golden set: document kinds, chains, critical values, consolidation, clusters."""

from __future__ import annotations

import json
import re
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path

from pydantic import BaseModel

from contract_analyst.eval.golden import GoldenRate, GoldenSet
from contract_analyst.ingest.normalize import normalize_text
from contract_analyst.kb.lemmatize import lemmatize
from contract_analyst.kb.store import KnowledgeBase
from contract_analyst.models import ConsolidatedContract, DocumentRecord, Extracted, ValueStatus
from contract_analyst.parsing.patterns import DOC_NUMBER_RE, find_dates

SCALAR_FIELDS = {"number": "number", "doc_date": "doc_date", "effective_from": "effective_from"}

_ISO_DATE = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")
_YEAR_WORD = r"(?:\s*(?:г\.|года\b|год\b|г\b))?"
_ZERO_KOPECKS = re.compile(r"(?<!\d)00?\s*коп\w*\.?")
_ENUMERATOR = re.compile(r"^\s*\d+(?:\.\d+)*\.\s")  # «2.2. Оплата …»: the clause number is not part of the value
_NUMBER = re.compile(r"(?<![\d.,])\d{1,3}(?:[   ]\d{3})+(?:[.,]\d+)?(?!\d)|\d+(?:[.,]\d+)*")
_TOKEN = re.compile(r"[a-zа-я]+(?:-[a-zа-я]+)*")
_STOP = {"для", "при", "или", "что", "чем", "как", "это", "этот", "тот", "такой", "также", "который", "свой", "весь",
         "его", "она", "они", "оно"}
_CANON = {"ежедневный": "ежедневно", "еженедельный": "еженедельно", "ежемесячный": "ежемесячно",
          "ежеквартальный": "ежеквартально", "ежегодный": "ежегодно", "ранее": "ранний", "руб": "рубль"}
# Mutually exclusive alternatives: a different member unmatched on each side is a substitution, not a paraphrase,
# however many other words the two texts share.
_RIVALS = (
    {"ежедневно", "еженедельно", "ежемесячно", "ежеквартально", "ежегодно"},
    {"день", "сутки", "неделя", "месяц", "квартал", "полугодие", "год"},
    {"рабочий", "календарный", "банковский"},
    {"менее", "более"},
    {"поздний", "ранний"},
    {"рубль", "доллар", "евро"},
    {"арендатор", "арендодатель", "субарендатор", "заказчик", "исполнитель", "подрядчик", "покупатель", "продавец",
     "поставщик", "дилер", "агент", "принципал", "лицензиар", "лицензиат", "компания"},
)
_MIN_WORD_COVERAGE = 0.6
_APPENDIX = re.compile(r"прил\w*\.?\s*(?:№\s*)?(\d+)")
_CLAUSE_REF = re.compile(r"\d+(?:\.\d+)*(?:\s*[-–—]\s*\d+(?:\.\d+)*)?")


class Scores(BaseModel):
    doc_kind_acc: float = 0.0
    chain_acc: float = 0.0
    critical_total: int = 0
    critical_accepted_correct: int = 0
    critical_accepted_wrong: int = 0
    critical_coverage: float = 0.0
    consolidation_acc: float = 0.0
    clusters_match: bool = False
    traps_passed: int = 0
    traps_total: int = 0
    wall_seconds: float = 0.0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    json_valid_rate: float = 1.0
    tool_call_valid_rate: float = 1.0
    safety_net: list[str] = []


def _norm_text(value) -> str:
    return " ".join(normalize_text(str(value)).casefold().replace("ё", "е").split())


def same_value(a, b) -> bool:
    if a is None or b is None:
        return a is b
    try:
        return Decimal(str(a).replace(",", ".")) == Decimal(str(b).replace(",", "."))
    except InvalidOperation:
        return _norm_text(a) == _norm_text(b)


def _facts_and_words(text: str | None) -> tuple[set[tuple], set[str]]:
    """Document numbers, dates and numbers as exact facts; the remaining words as lemmas."""
    facts: set[tuple] = set()
    t = normalize_text(str(text or ""))
    facts.update(("doc", n) for n in DOC_NUMBER_RE.findall(t))
    t = DOC_NUMBER_RE.sub(" ", t).casefold().replace("ё", "е")

    def iso(m: re.Match) -> str:
        try:
            facts.add(("date", date(int(m[1]), int(m[2]), int(m[3]))))
            return " "
        except ValueError:
            return m[0]

    t = _ISO_DATE.sub(iso, t)
    for d, span in find_dates(t):  # «31» декабря 2026 г. and 31.12.2026 are the same fact
        facts.add(("date", d))
        t = re.sub(re.escape(span) + _YEAR_WORD, " ", t, count=1)

    def number(m: re.Match) -> str:
        raw = re.sub(r"[   ]", "", m[0]).replace(",", ".")
        try:
            facts.add(("num", Decimal(raw)))
        except InvalidOperation:
            facts.add(("num", raw))  # a clause path like 1.1.1
        return " "

    t = _NUMBER.sub(number, _ZERO_KOPECKS.sub(" ", _ENUMERATOR.sub(" ", t)))
    words = set()
    for token in _TOKEN.findall(t):
        if len(token) >= 3:
            lemma = lemmatize(token).replace("ё", "е")
            words.add(_CANON.get(lemma, lemma))
    return facts, words - _STOP


def _related(a: str, b: str) -> bool:
    """Same word up to the ending: отказ ~ отказаться, письменный ~ письменно, but not ежемесячно ~ ежеквартально."""
    n = min(6, len(a), len(b))
    return a[:n] == b[:n]


def similar_text(a: str | None, b: str | None) -> bool:
    """Two wordings of one term value: the same numbers, dates and document numbers, compatible words.

    Words may be added, dropped or inflected (a shortened golden value against a longer extracted one), but a member
    of a group of alternatives (ежемесячно / ежеквартально, рабочих / календарных дней, Арендатор / Арендодатель)
    replaced by another member makes the texts different.
    """
    facts_a, words_a = _facts_and_words(a)
    facts_b, words_b = _facts_and_words(b)
    if facts_a != facts_b:
        return False
    if not (facts_a or words_a) or not (facts_b or words_b):
        return not (facts_a or words_a or facts_b or words_b)
    only_a = {w for w in words_a if not any(_related(w, x) for x in words_b)}
    only_b = {w for w in words_b if not any(_related(w, x) for x in words_a)}
    if any(only_a & group and only_b & group for group in _RIVALS):
        return False
    short, unmatched = (words_a, only_a) if len(words_a) <= len(words_b) else (words_b, only_b)
    return not short or 1 - len(unmatched) / len(short) >= _MIN_WORD_COVERAGE


def _accepted(ext: Extracted):
    if ext.status != ValueStatus.ACCEPTED or ext.value is None:
        return None
    return ext.value.isoformat() if isinstance(ext.value, date) else ext.value


def _accepted_rows(rows: list) -> list:
    """Only accepted rows count: a fold's conflict or rejected value is not an answer."""
    return [r for r in rows if r.status == ValueStatus.ACCEPTED]


def _rate_set(rates) -> set[tuple]:
    return {(Decimal(r.amount).normalize(), r.vat, r.valid_from, r.valid_to) for r in rates}


def normalize_target(target: str | None) -> str | None:
    """One spelling of what an operation changes: «п. 12.2», «пункт 12.2» and «12.2» are «12.2»; «Приложение № 1,
    п. 1.6» and «п. 1.6 Приложения 1» are «приложение 1 1.6»; a document is its number; text without a reference
    (for example «весь договор») is None."""
    if not target:
        return None
    t = normalize_text(target)
    docs = sorted(set(DOC_NUMBER_RE.findall(t)))
    t = DOC_NUMBER_RE.sub(" ", t).casefold()
    for _, span in find_dates(t):
        t = t.replace(span, " ", 1)
    appendices = sorted(f"приложение {n}" for n in _APPENDIX.findall(t))
    refs = sorted(re.sub(r"\s*[-–—]\s*", "-", r) for r in _CLAUSE_REF.findall(_APPENDIX.sub(" ", t)))
    return " ".join([*docs, *appendices, *refs]) or None


def _effective(*values) -> date | None:
    """The date a document's operations apply from when they name none: effective_from, else the document date."""
    for v in values:
        if v.status == ValueStatus.ACCEPTED and v.value:
            try:
                return v.value if isinstance(v.value, date) else date.fromisoformat(str(v.value))
            except ValueError:
                return None
    return None


def _op_set(ops, effective: date | None) -> set[tuple]:
    return {(str(o.op), normalize_target(o.target), o.effective_from or effective) for o in ops}


class _Tally:
    def __init__(self) -> None:
        self.total = self.correct = self.wrong = 0

    def scalar(self, golden_value, golden_status: str, predicted) -> None:
        if golden_status == "accepted" and golden_value is not None:
            self.total += 1
            if predicted is not None:
                if same_value(golden_value, predicted):
                    self.correct += 1
                else:
                    self.wrong += 1
        elif predicted is not None:
            self.wrong += 1  # value accepted where the golden set has none: a hallucination


def _critical(golden: GoldenSet, records: dict[str, DocumentRecord]) -> _Tally:
    tally = _Tally()
    for g in golden.documents:
        r = records.get(g.doc_id)
        for field in SCALAR_FIELDS:
            gv = getattr(g, field)
            tally.scalar(gv.value, gv.status, _accepted(getattr(r, field)) if r else None)
        parent = r.parent.value if r and r.parent.status == ValueStatus.ACCEPTED else None
        tally.scalar(g.parent_number.value, g.parent_number.status, parent.number if parent else None)
        tally.scalar(g.parent_date.value, g.parent_date.status,
                     parent.date_stated.isoformat() if parent and parent.date_stated else None)
        gold_rates = _rate_set(g.rates)
        got_rates = _rate_set([x for x in (r.rates if r else []) if x.status == ValueStatus.ACCEPTED])
        tally.total += len(gold_rates)
        tally.correct += len(gold_rates & got_rates)
        tally.wrong += len(got_rates - gold_rates)
        gold_ops = _op_set(g.ops, _effective(g.effective_from, g.doc_date))
        got_ops = _op_set([o for o in (r.ops if r else []) if o.status == ValueStatus.ACCEPTED],
                          _effective(r.effective_from, r.doc_date) if r else None)
        tally.total += len(gold_ops)
        tally.correct += len(gold_ops & got_ops)
        tally.wrong += len(got_ops - gold_ops)
    return tally


def _own_as_of(run_dir: Path) -> date | None:
    try:
        return date.fromisoformat(json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))["as_of"])
    except (OSError, KeyError, ValueError):
        return None


class RunsAt:
    """Where each golden date is scored from: the run made for that date, else the primary run.

    On its own date a run is judged by what it folded itself; only other dates are recomputed by the code fold,
    which is a fair answer for a code-fold run and nobody else's.
    """

    def __init__(self, primary: Path, runs_at: dict[date, Path] | None = None) -> None:
        self.primary = Path(primary)
        self.runs_at = {when: Path(p) for when, p in (runs_at or {}).items()}

    def terms(self, number: str, when: date) -> ConsolidatedContract | None:
        run = self.runs_at.get(when, self.primary)
        kb = KnowledgeBase(run / "kb")
        if _own_as_of(run) != when:
            return kb.get_contract_terms(number, when)
        stored = kb.consolidated().get(number)
        return stored if stored is not None and stored.as_of == when else None


def _consolidation(golden: GoldenSet, runs: RunsAt) -> float:
    points = earned = 0
    for chain in golden.chains:
        for when in golden.as_of_dates:
            result = runs.terms(chain.contract_number, when)
            if when in chain.status_at:
                points += 1
                earned += bool(result and result.status == chain.status_at[when])
            if when in chain.rates_at:
                points += 1
                earned += bool(result and _rate_set(_accepted_rows(result.rates)) == _rate_set(chain.rates_at[when]))
            for term, value in chain.terms_at.get(when, {}).items():
                points += 1
                state = result.terms.get(term) if result else None
                accepted = state is not None and state.current.status == ValueStatus.ACCEPTED
                earned += accepted and similar_text(value, state.current.value)
    return earned / points if points else 1.0


def _usage(run_dir: Path) -> dict:
    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    trace_path = run_dir / "trace.jsonl"
    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()] if trace_path.exists() else []
    calls = sum(1 for e in events if e.get("event") == "tool_call")
    bad = sum(1 for e in events if e.get("event") == "error" and e.get("message") == "bad arguments")
    metrics["tool_call_valid_rate"] = 1 - bad / (calls + bad) if calls + bad else 1.0
    return metrics


def latest_full_run(runs_dir: Path) -> Path | None:
    """The newest finished run over the whole package. metrics.json is written last, so a run without it is still
    going or crashed; its scope says whether --chain or --doc limited the run (older runs did not record it)."""
    for run in sorted(Path(runs_dir).glob("*"), reverse=True):
        try:
            scope = json.loads((run / "metrics.json").read_text(encoding="utf-8")).get("scope")
        except (OSError, ValueError):
            continue
        if (run / "kb").is_dir() and scope == {"chain": None, "doc": None}:
            return run
    return None


def score_run(run_dir: Path, golden: GoldenSet, runs_at: dict[date, Path] | None = None) -> Scores:
    """Score a run; runs_at gives the run made for each golden date (an experiment runs every variant per date)."""
    from contract_analyst.eval.traps import check_traps

    kb = KnowledgeBase(Path(run_dir) / "kb")
    records, chains = kb.records(), {c.contract_number: c for c in kb.chains()}
    kinds = [records.get(g.doc_id) is not None and records[g.doc_id].kind.value == g.kind for g in golden.documents]
    chain_ok = []
    for g in golden.chains:
        c = chains.get(g.contract_number)
        chain_ok.append(bool(c and c.main_doc_id == g.main_doc_id and [d.doc_id for d in c.docs] == g.order))
    tally = _critical(golden, records)
    clusters = {(c.subject_category, frozenset(c.members)) for c in kb.clusters()}
    traps = check_traps(Path(run_dir), golden, runs_at)
    usage = _usage(Path(run_dir))
    llm = usage.get("llm_usage", {})
    invalid = sum(1 for r in records.values() for i in r.issues if i.code == "llm_invalid_json")
    return Scores(
        doc_kind_acc=sum(kinds) / len(kinds) if kinds else 1.0,
        chain_acc=sum(chain_ok) / len(chain_ok) if chain_ok else 1.0,
        critical_total=tally.total, critical_accepted_correct=tally.correct, critical_accepted_wrong=tally.wrong,
        critical_coverage=tally.correct / tally.total if tally.total else 1.0,
        consolidation_acc=_consolidation(golden, RunsAt(run_dir, runs_at)),
        clusters_match=clusters == {(c.category, frozenset(c.members)) for c in golden.clusters},
        traps_passed=sum(t.passed for t in traps), traps_total=len(traps),
        wall_seconds=usage.get("seconds", 0.0), llm_calls=llm.get("calls", 0),
        input_tokens=llm.get("input_tokens", 0), output_tokens=llm.get("output_tokens", 0),
        json_valid_rate=1 - invalid / llm["calls"] if llm.get("calls") else 1.0,
        tool_call_valid_rate=usage["tool_call_valid_rate"], safety_net=usage.get("safety_net", []))


__all__ = ["GoldenRate", "RunsAt", "Scores", "latest_full_run", "normalize_target", "same_value", "score_run",
           "similar_text"]
