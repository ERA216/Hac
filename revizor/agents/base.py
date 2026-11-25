"""Базовый класс агента: шаблон промпта (Jinja2) + схема ответа + вызов LLM."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from pydantic import BaseModel

from ..criteria import SEVERITY_GUIDE
from ..llm import LLMClient

PROMPTS = Path(__file__).resolve().parent.parent / "prompts"
_env = Environment(loader=FileSystemLoader(PROMPTS), autoescape=False, undefined=StrictUndefined)


class Agent:
    name: str = "agent"
    title: str = "Агент"
    template: str = ""
    schema: type[BaseModel] = BaseModel

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def render(self, **ctx) -> tuple[str, str]:
        text = _env.get_template(self.template).render(severity_guide=SEVERITY_GUIDE, **ctx)
        system, user = text.split("=== USER ===", 1)
        return system.strip(), user.strip()

    def call(self, **ctx):
        system, user = self.render(**ctx)
        return self.llm.generate(self.name, system, user, self.schema)
