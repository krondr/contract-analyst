"""Clusters by subject category plus code-computed facets, and differences inside each cluster."""

from __future__ import annotations

from collections import Counter

from pydantic import BaseModel

from contract_analyst.config import Settings
from contract_analyst.llm.client import LLMClient
from contract_analyst.llm.prompts import load_prompt
from contract_analyst.llm.structured import StructuredFailure, call_structured
from contract_analyst.models import (
    Cluster,
    ConsolidatedContract,
    ContractChain,
    Difference,
    DocumentRecord,
    ValueStatus,
)

NO_DATA = "нет данных"
STATUS_RU = {"active": "действует", "partially_suspended": "частично прекращён", "terminated": "расторгнут",
             "not_yet_in_force": "не вступил в силу", "unknown": "неизвестно"}
FACETS = ("counterparty_type", "city", "remuneration_model", "status", "year")


class SummaryOut(BaseModel):
    summary: str


def _ref(source) -> str:
    return f" [doc:{source.doc_id}:p{source.page}]" if source else ""


def compute_facets(c: ConsolidatedContract, main: DocumentRecord | None) -> dict[str, str]:
    counterparty = NO_DATA
    if main and main.parties:
        org = main.parties[-1].org_type
        counterparty = org.value if org.status == ValueStatus.ACCEPTED else (
            "вымарано" if org.status == ValueStatus.REDACTED else NO_DATA)
    labels = {r.label for r in [*c.rates, *c.planned_rates]}
    model = "ставки по группам" if len(labels) > 1 else "фиксированная сумма" if labels else NO_DATA
    city = main.city.value if main and main.city.status == ValueStatus.ACCEPTED else NO_DATA
    year = str(main.doc_date.value.year) if main and main.doc_date.status == ValueStatus.ACCEPTED else NO_DATA
    return {"counterparty_type": counterparty, "city": city, "remuneration_model": model,
            "status": STATUS_RU.get(c.status, c.status), "year": year}


def _category(chain: ContractChain, records: dict[str, DocumentRecord]) -> str:
    def cat(doc_id: str) -> str | None:
        r = records.get(doc_id)
        if r and r.subject_category.status == ValueStatus.ACCEPTED and r.subject_category.value != "other":
            return r.subject_category.value
        return None

    if chain.main_doc_id and (main_cat := cat(chain.main_doc_id)):
        return main_cat
    votes = Counter(c for d in chain.docs if (c := cat(d.doc_id)))
    return votes.most_common(1)[0][0] if votes else "other"


def _amount(rate) -> str:
    """A rate's amount with its unit, so that 60 % and 60 руб. are not the same value."""
    return f"{format(rate.amount.normalize(), 'f')} {'%' if rate.unit == '%' else 'руб.'}"


def compare_cluster(cluster: Cluster, contracts: dict[str, ConsolidatedContract]) -> list[Difference]:
    members = [m for m in cluster.members if m in contracts]
    if len(members) < 2:
        return []
    diffs: list[Difference] = []
    for facet in FACETS:
        values = {m: cluster.facets.get(m, {}).get(facet, NO_DATA) for m in members}
        if len(set(values.values())) > 1:
            diffs.append(Difference(term=facet, values=values))
    for name in sorted({t for m in members for t in contracts[m].terms}):
        values = {}
        for m in members:
            state = contracts[m].terms.get(name)
            values[m] = state.current.value if state and state.current.status == ValueStatus.ACCEPTED else NO_DATA
        if len(set(values.values())) > 1:
            diffs.append(Difference(term=name, values=values))
    for label in sorted({r.label for m in members for r in contracts[m].rates}):
        values = {m: next((_amount(r) for r in contracts[m].rates if r.label == label), NO_DATA) for m in members}
        if len(set(values.values())) > 1:
            diffs.append(Difference(term=f"rate:{label}", values=values))
    return diffs


def cluster_contracts(contracts: list[ConsolidatedContract], records: dict[str, DocumentRecord],
                      chains: list[ContractChain], settings: Settings) -> list[Cluster]:
    by_number = {c.contract_number: c for c in contracts}
    groups: dict[str, list[ContractChain]] = {}
    for chain in chains:
        if chain.contract_number in by_number:
            groups.setdefault(_category(chain, records), []).append(chain)
    clusters = []
    for i, (category, members) in enumerate(sorted(groups.items()), start=1):
        facets = {ch.contract_number: compute_facets(by_number[ch.contract_number], records.get(ch.main_doc_id or ""))
                  for ch in members}
        cluster = Cluster(cluster_id=f"c{i}", label=settings.subject_categories.get(category, category),
                          subject_category=category, members=sorted(facets), facets=facets)
        cluster.differences = compare_cluster(cluster, by_number)
        clusters.append(cluster)
    return clusters


def _cluster_facts(cluster: Cluster, contracts: dict[str, ConsolidatedContract]) -> str:
    lines = [f"Кластер: {cluster.label}"]
    for m in cluster.members:
        c = contracts.get(m)
        if c is None:
            continue
        facets = ", ".join(f"{k}: {v}" for k, v in cluster.facets.get(m, {}).items())
        lines.append(f"- Договор {m}: {facets}")
        for name, state in c.terms.items():
            if state.current.status == ValueStatus.ACCEPTED:
                lines.append(f"  - {name}: {state.current.value}{_ref(state.current.source)}")
        for r in c.rates:
            lines.append(f"  - ставка «{r.label}»: {_amount(r)}{_ref(r.source)}")
    return "\n".join(lines)


def summarize_cluster(cluster: Cluster, contracts: dict[str, ConsolidatedContract], client: LLMClient) -> str | None:
    prompt = load_prompt("cluster_summary")
    messages = [{"role": "system", "content": prompt.render()},
                {"role": "user", "content": _cluster_facts(cluster, contracts)}]
    result = call_structured(client, messages, SummaryOut)
    return None if isinstance(result, StructuredFailure) else result.summary.strip()
