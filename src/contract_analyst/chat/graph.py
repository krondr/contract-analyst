"""Chat over the knowledge base: LangGraph assistant/tools loop, SQLite-checkpointed memory, verified citations."""

from __future__ import annotations

import operator
import re
import sqlite3
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Overwrite

from contract_analyst.chat.tools import chat_tool_schemas, run_chat_tool
from contract_analyst.config import Settings
from contract_analyst.kb.store import KnowledgeBase
from contract_analyst.llm.client import LLMClient, Message, ToolCall
from contract_analyst.llm.prompts import load_prompt
from contract_analyst.report.verify import verify_text

REFUSAL = "Нет данных в базе знаний."
MAX_TOOL_ROUNDS = 6


class ChatState(TypedDict):
    messages: Annotated[list[Message], operator.add]
    rounds: int


class ChatSession:
    def __init__(self, kb: KnowledgeBase, client: LLMClient, settings: Settings, memory_path: Path):
        self.kb, self.client, self.settings = kb, client, settings
        memory_path.parent.mkdir(parents=True, exist_ok=True)
        self._saver = SqliteSaver(sqlite3.connect(memory_path, check_same_thread=False))
        self._system = load_prompt("chat").render(contracts=[c["contract_number"] for c in kb.list_contracts()])
        self._sources = kb.sources()
        self.graph = self._build()

    def _build(self):
        def assistant(state: ChatState) -> dict:
            turn = self.client.chat([{"role": "system", "content": self._system}, *state["messages"]], chat_tool_schemas())
            calls = [c.model_dump() for c in turn.tool_calls] if state.get("rounds", 0) < MAX_TOOL_ROUNDS else []
            return {"messages": [{"role": "assistant", "content": turn.content, "tool_calls": calls}]}

        def tools(state: ChatState) -> dict:
            out = []
            for raw in state["messages"][-1]["tool_calls"]:
                call = ToolCall.model_validate(raw)
                out.append({"role": "tool", "tool_call_id": call.id, "name": call.name,
                            "content": run_chat_tool(self.kb, call.name, call.args)})
            return {"messages": out, "rounds": state.get("rounds", 0) + 1}

        def answer(state: ChatState) -> dict:
            """Verify the final answer and collapse the turn in memory to the question and what the user saw,
            so a removed claim or a raw tool result cannot leak into the next question."""
            messages = state["messages"]
            start = max(i for i, m in enumerate(messages) if m["role"] == "user")
            text, _ = verify_text(messages[-1].get("content") or "", self._sources, require_refs=True,
                                  question=messages[start]["content"])
            reply = {"role": "assistant", "content": text if re.search(r"\w", text) else REFUSAL}
            return {"messages": Overwrite([*messages[:start + 1], reply])}

        graph = StateGraph(ChatState)
        graph.add_node("assistant", assistant)
        graph.add_node("tools", tools)
        graph.add_node("answer", answer)
        graph.add_edge(START, "assistant")
        graph.add_conditional_edges("assistant", lambda s: "tools" if s["messages"][-1].get("tool_calls") else "answer",
                                    ["tools", "answer"])
        graph.add_edge("tools", "assistant")
        graph.add_edge("answer", END)
        return graph.compile(checkpointer=self._saver)

    def ask(self, question: str, thread: str) -> str:
        state = self.graph.invoke({"messages": [{"role": "user", "content": question}], "rounds": 0},
                                  config={"configurable": {"thread_id": thread}, "recursion_limit": 40})
        return state["messages"][-1]["content"]
