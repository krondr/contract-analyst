"""Versioned prompt templates: prompts/<name>_v<N>.md, highest N wins."""

from __future__ import annotations

import re
from pathlib import Path

import jinja2
from pydantic import BaseModel

from contract_analyst.config import REPO_ROOT

PROMPTS_DIR = REPO_ROOT / "prompts"
_ENV = jinja2.Environment(undefined=jinja2.StrictUndefined, keep_trailing_newline=False, autoescape=False)


class Prompt(BaseModel):
    name: str
    version: int
    template: str

    def render(self, **variables) -> str:
        return _ENV.from_string(self.template).render(**variables)


def load_prompt(name: str, prompts_dir: Path | None = None) -> Prompt:
    directory = prompts_dir or PROMPTS_DIR
    versions = {}
    for path in directory.glob(f"{name}_v*.md"):
        if m := re.fullmatch(rf"{re.escape(name)}_v(\d+)\.md", path.name):
            versions[int(m.group(1))] = path
    if not versions:
        raise FileNotFoundError(f"no prompt {name}_v*.md in {directory}")
    version = max(versions)
    return Prompt(name=name, version=version, template=versions[version].read_text(encoding="utf-8"))
