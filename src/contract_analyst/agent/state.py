"""LangGraph state and the structured answers of the planner and the reflector."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from pydantic import BaseModel

from contract_analyst.llm.client import Message


class AgentState(TypedDict):
    messages: Annotated[list[Message], operator.add]
    plan: list[str]
    reflection_round: int
    gaps: list[str]


class PlanOut(BaseModel):
    items: list[str]


class ReflectionOut(BaseModel):
    goal_met: bool
    gaps: list[str] = []
