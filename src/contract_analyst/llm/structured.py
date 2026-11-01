"""Schema-validated LLM calls with corrective retries."""

from __future__ import annotations

import re

from pydantic import BaseModel, ValidationError

from contract_analyst.llm.client import LLMClient, Message

_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)
SHORTER = ("Предыдущий ответ оборвался на лимите длины. Ответь заново короче: каждый элемент перечисли один раз, "
           "без повторов; цитату приводи дословно, но только нужную фразу.")


class StructuredFailure(BaseModel):
    errors: list[str]
    raw: list[str]
    truncated: bool = False  # the answer kept hitting the token cap


def _strip_fences(raw: str) -> str:
    m = _FENCE.match(raw)
    return m.group(1) if m else raw


def _cut_off(raw: str, exc: ValidationError) -> bool:
    """The JSON stops mid-value. Answers are decoded against the schema, so that only happens at the token cap;
    the text alone tells, so a replayed run decides the same way."""
    return bool(raw.strip()) and any(e["type"] == "json_invalid" and "EOF" in e["msg"] for e in exc.errors())


def call_structured[T: BaseModel](client: LLMClient, messages: list[Message], schema: type[T],
                                  retries: int = 2) -> T | StructuredFailure:
    msgs = list(messages)
    errors: list[str] = []
    raws: list[str] = []
    cut = shortened = False
    for _ in range(retries + 1):
        raw = client.structured(msgs, schema)
        raws.append(raw)
        try:
            return schema.model_validate_json(_strip_fences(raw))
        except ValidationError as exc:
            error, cut = str(exc)[:500], _cut_off(raw, exc)
        errors.append(error)
        if cut:
            if shortened:
                break
            # The same request at temperature 0 would run into the cap the same way, and the cut text would only
            # eat the context: ask once more, from the original messages, for a shorter answer.
            msgs, shortened = list(messages) + [{"role": "user", "content": SHORTER}], True
            continue
        msgs = msgs + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": f"Ответ не прошёл проверку схемы: {error}. Верни только JSON по схеме."},
        ]
    return StructuredFailure(errors=errors, raw=raws, truncated=cut)
