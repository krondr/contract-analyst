"""Shared data models: the contract between ingest, understanding, assembly, reporting and the agent."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel


class PageMethod(StrEnum):
    TEXT = "text"
    OCR = "ocr"
    EMPTY = "empty"


class DocFormat(StrEnum):
    PDF = "pdf"
    DOCX = "docx"
    IMAGE = "image"
    UNSUPPORTED = "unsupported"


class DocKind(StrEnum):
    MAIN = "main"
    SUPPLEMENT = "supplement"
    FRAMEWORK = "framework"
    UNKNOWN = "unknown"


class ValueStatus(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    CONFLICT = "conflict"
    REDACTED = "redacted"
    ABSENT = "absent"


class OpType(StrEnum):
    REPLACE_CLAUSE = "replace_clause"
    ADD_CLAUSE = "add_clause"
    DELETE_CLAUSE = "delete_clause"
    RESTATE_DOCUMENT = "restate_document"
    RESTATE_SECTION = "restate_section"
    REPLACE_APPENDIX = "replace_appendix"
    TERMINATE = "terminate"
    SUSPEND_PART = "suspend_part"
    INVALIDATE_DOCUMENT = "invalidate_document"
    OTHER = "other"


class OcrWord(BaseModel):
    text: str
    conf: float  # 0..1


class PageQuality(BaseModel):
    chars: int
    ocr_mean_conf: float | None = None
    low_conf_ratio: float | None = None
    rotation: int = 0


class Page(BaseModel):
    number: int  # 1-based
    method: PageMethod
    text: str
    quality: PageQuality
    words: list[OcrWord] = []


class InputFile(BaseModel):
    path: str
    name: str
    sha256: str
    format: DocFormat
    size_bytes: int
    duplicate_of: str | None = None


class Clause(BaseModel):
    clause_id: str  # "preamble", "1.25", "12.2", "Приложение 7"
    page: int
    text: str


class DocumentText(BaseModel):
    doc_id: str
    file: InputFile
    pages: list[Page]
    clauses: list[Clause]
    status: Literal["ok", "low_quality", "empty", "error"]
    issues: list[str] = []


class Source(BaseModel):
    doc_id: str
    page: int
    quote: str
    clause_id: str | None = None


class Extracted[T](BaseModel):
    value: T | None = None
    status: ValueStatus
    source: Source | None = None
    confidence: float | None = None
    reason: str | None = None
    alternatives: list[Source] = []


class ParentRef(BaseModel):
    number: str | None = None
    date_stated: date | None = None


class Party(BaseModel):
    role: str
    org_type: Extracted[str]
    name: Extracted[str]


class RateRow(BaseModel):
    label: str
    amount: Decimal
    unit: str = "RUB"
    vat: Literal["included", "excluded", "unknown"] = "unknown"
    valid_from: date | None = None
    valid_to: date | None = None
    status: ValueStatus
    source: Source | None = None
    reason: str | None = None


class AmendmentOp(BaseModel):
    op: OpType
    target: str | None = None
    new_text: str | None = None
    effective_from: date | None = None
    effective_to: date | None = None
    status: ValueStatus
    source: Source | None = None
    reason: str | None = None


class Issue(BaseModel):
    code: str
    message: str
    sources: list[Source] = []


class DocumentRecord(BaseModel):
    doc_id: str
    file_name: str
    title: str
    is_draft: bool = False
    kind: Extracted[DocKind]
    number: Extracted[str]
    doc_date: Extracted[date]
    city: Extracted[str]
    parent: Extracted[ParentRef]
    effective_from: Extracted[date]
    subject_category: Extracted[str]
    parties: list[Party] = []
    terms: dict[str, Extracted[str]] = {}
    rates: list[RateRow] = []
    ops: list[AmendmentOp] = []
    issues: list[Issue] = []


class ChainDoc(BaseModel):
    doc_id: str
    number: str | None
    role: Literal["main", "supplement", "unknown"]
    order_key: str
    order_basis: Literal["effective_from", "doc_date", "number"]
    invalidated_by: str | None = None


class ContractChain(BaseModel):
    contract_number: str
    main_doc_id: str | None
    main_date: Extracted[date] | None = None
    docs: list[ChainDoc]
    issues: list[Issue] = []


class TermChange(BaseModel):
    value: str
    effective_from: date | None
    doc_id: str
    source: Source | None = None
    op: OpType | None = None


class TermState(BaseModel):
    name: str
    current: Extracted[str]
    history: list[TermChange] = []
    planned: list[TermChange] = []


class TimelineEvent(BaseModel):
    when: date | None
    doc_id: str
    number: str | None
    summary: str
    invalidated: bool = False


class ConsolidatedContract(BaseModel):
    contract_number: str
    as_of: date
    status: Literal["active", "not_yet_in_force", "partially_suspended", "terminated", "unknown"]
    status_since: date | None = None
    terms: dict[str, TermState] = {}
    rates: list[RateRow] = []
    planned_rates: list[RateRow] = []
    timeline: list[TimelineEvent] = []
    issues: list[Issue] = []


class Difference(BaseModel):
    term: str
    values: dict[str, str]  # contract_number -> value


class Cluster(BaseModel):
    cluster_id: str
    label: str
    subject_category: str
    members: list[str]
    facets: dict[str, dict[str, str]] = {}
    differences: list[Difference] = []
    summary: str | None = None


class Decision(BaseModel):
    issue_code: str
    contract_number: str | None
    decision: str
    rationale: str


class ToolResult(BaseModel):
    status: Literal["ok", "empty", "error", "low_quality"]
    data: dict = {}
    error_type: str | None = None
    hint: str | None = None
