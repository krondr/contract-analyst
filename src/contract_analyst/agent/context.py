"""Mutable state of one analysis run, shared by graph nodes and tools."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from contract_analyst.agent.state import PlanCheck
from contract_analyst.agent.trace import TraceLogger
from contract_analyst.config import Settings
from contract_analyst.llm.client import LLMClient
from contract_analyst.models import (
    Cluster,
    ConsolidatedContract,
    ContractChain,
    Decision,
    DocumentRecord,
    DocumentText,
    InputFile,
)
from contract_analyst.ocr import OcrEngine
from contract_analyst.parsing.header import HeaderGuess
from contract_analyst.report.render import ReflectionItem

PARTS = ("documents", "records", "chains", "consolidated", "clusters", "decisions")
# What each derived part is built from. A part built from older inputs is stale: the repeat guard lets the agent
# rebuild it, and the safety net rebuilds it (and so, in turn, everything built from it) before the report.
INPUTS = {"chains": ("records",), "consolidated": ("records", "chains"),
          "clusters": ("records", "chains", "consolidated")}


@dataclass
class RunContext:
    settings: Settings
    client: LLMClient
    ocr_engine: OcrEngine
    trace: TraceLogger
    input_dir: Path
    run_dir: Path
    as_of: date
    inventory: list[InputFile] = field(default_factory=list)
    documents: dict[str, DocumentText] = field(default_factory=dict)
    headers: dict[str, HeaderGuess] = field(default_factory=dict)
    records: dict[str, DocumentRecord] = field(default_factory=dict)
    analyzed_groups: dict[str, set[str]] = field(default_factory=dict)  # field groups each record went through
    chains: list[ContractChain] = field(default_factory=list)
    consolidated: dict[str, ConsolidatedContract] = field(default_factory=dict)
    clusters: list[Cluster] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)
    unprocessed: list[dict] = field(default_factory=list)
    reflection: list[ReflectionItem] = field(default_factory=list)
    plan: list[str] = field(default_factory=list)  # the planner's own items, without the mandatory ones
    plan_checks: dict[int, PlanCheck] = field(default_factory=dict)  # the reflector's verdict by item number
    only_doc_ids: set[str] | None = None
    only_paths: set[str] | None = None
    safety_net: list[str] = field(default_factory=list)
    agent_reads: bool = False
    read_cache: Path | None = None
    report_path: Path | None = None
    removed_claims: int = 0
    revision: Counter[str] = field(default_factory=Counter)  # per part of PARTS, bumped on every change
    built_from: dict[str, tuple[int, ...]] = field(default_factory=dict)  # derived part -> revisions of its INPUTS

    def changed(self, part: str) -> None:
        self.revision[part] += 1

    def revisions(self, parts: tuple[str, ...]) -> tuple[int, ...]:
        return tuple(self.revision[p] for p in parts)

    def built(self, part: str) -> None:
        self.built_from[part] = self.revisions(INPUTS[part])

    def stale(self, part: str) -> bool:
        return self.built_from.get(part) != self.revisions(INPUTS[part])
