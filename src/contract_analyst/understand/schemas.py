"""JSON schemas the LLM must answer with. Every value carries a page and a verbatim quote."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from contract_analyst.models import DocKind, OpType


class ValueOut(BaseModel):
    value: str | None = None
    page: int | None = None
    quote: str | None = None
    redacted: bool = False


class PartyOut(BaseModel):
    role: str
    org_type: ValueOut = ValueOut()
    name: ValueOut = ValueOut()


class HeaderOut(BaseModel):
    kind: DocKind
    title: str = ""
    number: ValueOut = ValueOut()
    doc_date: ValueOut = ValueOut()
    city: ValueOut = ValueOut()
    parent_number: ValueOut = ValueOut()
    parent_date: ValueOut = ValueOut()
    effective_from: ValueOut = ValueOut()
    parties: list[PartyOut] = []
    subject_category: str = "other"


class TermsOut(BaseModel):
    values: dict[str, ValueOut] = {}


class RateOut(BaseModel):
    label: str
    amount: str
    vat: Literal["included", "excluded", "unknown"] = "unknown"
    valid_from: str | None = None
    valid_to: str | None = None
    page: int
    quote: str


class RatesOut(BaseModel):
    rates: list[RateOut] = []


class OpOut(BaseModel):
    op: OpType
    target: str | None = None
    new_text_excerpt: str | None = None
    effective_from: str | None = None
    effective_to: str | None = None
    page: int
    quote: str


class OpsOut(BaseModel):
    ops: list[OpOut] = []
