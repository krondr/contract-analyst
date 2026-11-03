"""Knowledge base of one run: accepted facts as JSON plus an FTS5 index of clauses."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import date
from pathlib import Path

from pydantic import BaseModel, TypeAdapter

from contract_analyst.assemble.consolidate import consolidate
from contract_analyst.kb.lemmatize import lemmatize
from contract_analyst.models import (
    Cluster,
    ConsolidatedContract,
    ContractChain,
    Decision,
    DocumentRecord,
    DocumentText,
    Extracted,
    Issue,
    TermChange,
    ValueStatus,
)
from contract_analyst.report.verify import grounding

_KEEP = (ValueStatus.ACCEPTED, ValueStatus.CONFLICT)


class ClauseHit(BaseModel):
    doc_id: str
    contract_number: str | None
    clause_id: str
    page: int
    snippet: str
    score: float


def _scrub(ext: Extracted) -> Extracted:
    """A value the gate did not keep leaves only its status and reason code: its quote and the detailed reason
    ("даты 2027-12-31 нет в цитате") can repeat the value itself."""
    if ext.status in _KEEP:
        return ext
    return type(ext)(status=ext.status, reason=ext.reason.split(":")[0] if ext.reason else None)


def _scrub_record(r: DocumentRecord) -> DocumentRecord:
    return r.model_copy(update={
        **{name: _scrub(v) for name in type(r).model_fields if isinstance(v := getattr(r, name), Extracted)},
        "terms": {k: _scrub(v) for k, v in r.terms.items()},
        "parties": [p.model_copy(update={"name": _scrub(p.name), "org_type": _scrub(p.org_type)}) for p in r.parties],
        "rates": [x for x in r.rates if x.status == ValueStatus.ACCEPTED],
        "ops": [x for x in r.ops if x.status == ValueStatus.ACCEPTED],
        "refused_ops": r.refused_ops + sum(x.status != ValueStatus.ACCEPTED for x in r.ops),
    })


class KnowledgeBase:
    def __init__(self, root: Path):
        self.root = Path(root)

    # --- write -------------------------------------------------------------------------------------------
    def _dump(self, name: str, adapter: TypeAdapter, value) -> None:
        (self.root / name).write_bytes(adapter.dump_json(value, indent=1))

    def save(self, documents: dict[str, DocumentText], records: dict[str, DocumentRecord],
             chains: list[ContractChain], consolidated: dict[str, ConsolidatedContract], clusters: list[Cluster],
             decisions: list[Decision]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self._dump("records.json", TypeAdapter(dict[str, DocumentRecord]),
                   {k: _scrub_record(v) for k, v in records.items()})
        self._dump("chains.json", TypeAdapter(list[ContractChain]), chains)
        self._dump("consolidated.json", TypeAdapter(dict[str, ConsolidatedContract]), consolidated)
        self._dump("clusters.json", TypeAdapter(list[Cluster]), clusters)
        self._dump("decisions.json", TypeAdapter(list[Decision]), decisions)
        pages = {doc_id: len(d.pages) for doc_id, d in documents.items()}
        names = {doc_id: d.file.name for doc_id, d in documents.items()}
        (self.root / "documents.json").write_text(json.dumps({"pages": pages, "names": names}, ensure_ascii=False),
                                                  encoding="utf-8")
        contract_of = {d.doc_id: c.contract_number for c in chains for d in c.docs}
        db_path = self.root / "kb.sqlite"
        db_path.unlink(missing_ok=True)
        with closing(sqlite3.connect(db_path)) as db, db:  # the connection's own context only commits
            db.execute("CREATE VIRTUAL TABLE clauses USING fts5(doc_id UNINDEXED, contract_number UNINDEXED, "
                       "clause_id UNINDEXED, page UNINDEXED, text, lemmas)")
            db.executemany("INSERT INTO clauses VALUES (?, ?, ?, ?, ?, ?)", [
                (doc_id, contract_of.get(doc_id), c.clause_id, c.page, c.text, lemmatize(c.text))
                for doc_id, d in documents.items() for c in d.clauses])
            # Whole pages, not clauses: a clause can run onto the next page, and the claim check needs page text.
            db.execute("CREATE TABLE pages (doc_id TEXT, page INTEGER, text TEXT, PRIMARY KEY (doc_id, page))")
            db.executemany("INSERT INTO pages VALUES (?, ?, ?)",
                           [(doc_id, p.number, p.text) for doc_id, d in documents.items() for p in d.pages])

    # --- read --------------------------------------------------------------------------------------------
    def _load(self, name: str, adapter: TypeAdapter, default):
        path = self.root / name
        return adapter.validate_json(path.read_bytes()) if path.exists() else default

    def records(self) -> dict[str, DocumentRecord]:
        return self._load("records.json", TypeAdapter(dict[str, DocumentRecord]), {})

    def chains(self) -> list[ContractChain]:
        return self._load("chains.json", TypeAdapter(list[ContractChain]), [])

    def consolidated(self) -> dict[str, ConsolidatedContract]:
        return self._load("consolidated.json", TypeAdapter(dict[str, ConsolidatedContract]), {})

    def clusters(self) -> list[Cluster]:
        return self._load("clusters.json", TypeAdapter(list[Cluster]), [])

    def list_contracts(self) -> list[dict]:
        cluster_of = {m: c.label for c in self.clusters() for m in c.members}
        consolidated = self.consolidated()
        return [{"contract_number": ch.contract_number, "status": consolidated[ch.contract_number].status
                 if ch.contract_number in consolidated else "unknown", "cluster": cluster_of.get(ch.contract_number),
                 "main_doc_id": ch.main_doc_id, "documents": len(ch.docs)} for ch in self.chains()]

    def get_contract_terms(self, number: str, as_of: date | None = None) -> ConsolidatedContract | None:
        stored = self.consolidated().get(number)
        if stored is not None and (as_of is None or as_of == stored.as_of):
            return stored
        chain = next((c for c in self.chains() if c.contract_number == number), None)
        return consolidate(chain, self.records(), as_of) if chain and as_of else stored

    def get_term_history(self, number: str, term: str) -> list[TermChange]:
        contract = self.get_contract_terms(number)
        state = contract.terms.get(term) if contract else None
        if state is None:
            return []
        current = [] if state.current.status not in _KEEP else [TermChange(
            value=state.current.value, effective_from=state.since,
            doc_id=state.current.source.doc_id if state.current.source else "", source=state.current.source)]
        return [*state.history, *current, *state.planned]

    def get_issues(self, number: str | None = None) -> list[Issue]:
        issues = [i for c in self.chains() if number in (None, c.contract_number) for i in c.issues]
        issues += [i for n, c in self.consolidated().items() if number in (None, n) for i in c.issues]
        return issues

    def search_clauses(self, query: str, contract: str | None = None, limit: int = 5) -> list[ClauseHit]:
        tokens = [t for t in lemmatize(query).split() if len(t) > 1]
        if not tokens or not (self.root / "kb.sqlite").exists():
            return []
        match = " OR ".join(f'"{t}"' for t in tokens)
        sql = ("SELECT doc_id, contract_number, clause_id, page, substr(text, 1, 300), bm25(clauses) FROM clauses "
               "WHERE lemmas MATCH ?" + (" AND contract_number = ?" if contract else "") + " ORDER BY bm25(clauses) LIMIT ?")
        params = [match, *([contract] if contract else []), limit]
        with closing(sqlite3.connect(self.root / "kb.sqlite")) as db:
            rows = db.execute(sql, params).fetchall()
        return [ClauseHit(doc_id=r[0], contract_number=r[1], clause_id=r[2], page=int(r[3]), snippet=r[4], score=-r[5])
                for r in rows]

    def sources(self) -> dict[str, str]:
        """Every valid reference with what can back a claim citing it: page text and the values from that page."""
        texts = {}
        path = self.root / "kb.sqlite"
        if path.exists():
            with closing(sqlite3.connect(path)) as db:
                if db.execute("SELECT 1 FROM sqlite_master WHERE name = 'pages'").fetchone():
                    texts = {(d, p): t for d, p, t in db.execute("SELECT doc_id, page, text FROM pages")}
        # A base saved before page texts were stored still has its references; only its values can back claims.
        pages = [(doc_id, page, texts.get((doc_id, page), "")) for doc_id, page in self._pages()]
        return grounding(pages, self.records(), self.chains(), self.consolidated())

    def _pages(self) -> list[tuple[str, int]]:
        path = self.root / "documents.json"
        counts = json.loads(path.read_text(encoding="utf-8"))["pages"] if path.exists() else {}
        return [(doc_id, p) for doc_id, n in counts.items() for p in range(1, n + 1)]

    def valid_refs(self) -> set[str]:
        return {f"doc:{doc_id}:p{page}" for doc_id, page in self._pages()}
