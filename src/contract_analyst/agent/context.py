"""Mutable state of one analysis run, shared by graph nodes and tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

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
    chains: list[ContractChain] = field(default_factory=list)
    consolidated: dict[str, ConsolidatedContract] = field(default_factory=dict)
    clusters: list[Cluster] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)
    unprocessed: list[dict] = field(default_factory=list)
    reflection: list[ReflectionItem] = field(default_factory=list)
    only_doc_ids: set[str] | None = None
    only_paths: set[str] | None = None
    safety_net: list[str] = field(default_factory=list)
    read_cache: Path | None = None
    report_path: Path | None = None
    removed_claims: int = 0
