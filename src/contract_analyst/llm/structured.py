"""Schema-validated LLM calls with corrective retries."""

from __future__ import annotations

import re

from pydantic import BaseModel, ValidationError

from contract_analyst.llm.client import LLMClient, Message

_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


class StructuredFailure(BaseModel):
    errors: list[str]
    raw: list[str]


def _strip_fences(raw: str) -> str:
    m = _FENCE.match(raw)
    return m.group(1) if m else raw


def call_structured[T: BaseModel](client: LLMClient, messages: list[Message], schema: type[T],
                                  retries: int = 2) -> T | StructuredFailure:
    msgs = list(messages)
    errors: list[str] = []
    raws: list[str] = []
    for _ in range(retries + 1):
        raw = client.structured(msgs, schema)
        raws.append(raw)
        try:
            return schema.model_validate_json(_strip_fences(raw))
        except ValidationError as exc:
            error = str(exc)[:500]
        errors.append(error)
        msgs = msgs + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": f"Ответ не прошёл проверку схемы: {error}. Верни только JSON по схеме."},
        ]
    return StructuredFailure(errors=errors, raw=raws)
