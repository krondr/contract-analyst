"""Golden set schema: hand-verified answers every experiment is scored against."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

from contract_analyst.models import DocKind, OpType, ValueStatus

TrapKind = Literal["issue_present", "status_is", "order_is", "excluded_from_consolidation", "term_value_at",
                   "rate_planned", "doc_kind_is", "normalized_number"]


class GoldenValue(BaseModel):
    value: str | None = None
    status: ValueStatus = ValueStatus.ACCEPTED
    page: int | None = None
    quote: str | None = None


class GoldenRate(BaseModel):
    label: str
    amount: Decimal
    vat: Literal["included", "excluded", "unknown"] = "unknown"
    valid_from: date | None = None
    valid_to: date | None = None
    page: int | None = None
    quote: str | None = None


class GoldenOp(BaseModel):
    op: OpType
    target: str | None = None
    effective_from: date | None = None
    page: int | None = None
    quote: str | None = None


class GoldenDocument(BaseModel):
    file: str
    doc_id: str
    kind: DocKind
    is_draft: bool = False
    number: GoldenValue = GoldenValue()
    doc_date: GoldenValue = GoldenValue()
    city: GoldenValue = GoldenValue()
    parent_number: GoldenValue = GoldenValue()
    parent_date: GoldenValue = GoldenValue()
    effective_from: GoldenValue = GoldenValue()
    subject_category: str = "other"
    counterparty_type: GoldenValue = GoldenValue()
    terms: dict[str, GoldenValue] = {}
    rates: list[GoldenRate] = []
    ops: list[GoldenOp] = []


class GoldenChain(BaseModel):
    contract_number: str
    main_doc_id: str | None = None
    order: list[str] = []
    invalidated: list[str] = []
    status_at: dict[date, str] = {}
    terms_at: dict[date, dict[str, str]] = {}
    rates_at: dict[date, list[GoldenRate]] = {}


class GoldenCluster(BaseModel):
    category: str
    members: list[str]


class GoldenTrap(BaseModel):
    trap_id: str
    kind: TrapKind
    contract_number: str | None = None
    doc_id: str | None = None
    expected: dict = {}


class GoldenScanPage(BaseModel):
    file: str
    page: int
    text: str


class GoldenSet(BaseModel):
    version: int = 1
    as_of_dates: list[date] = []
    documents: list[GoldenDocument] = []
    chains: list[GoldenChain] = []
    clusters: list[GoldenCluster] = []
    traps: list[GoldenTrap] = []
    scans: list[GoldenScanPage] = []


def load_golden(path: Path) -> GoldenSet:
    return GoldenSet.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})


def dump_golden(gs: GoldenSet, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(gs.model_dump(mode="json", exclude_defaults=False), allow_unicode=True,
                                         sort_keys=False, width=120), encoding="utf-8")
