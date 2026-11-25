"""Агенты «Ревизора ТЗ». У каждого — одна ответственность, свой промпт и схема ответа."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..criteria import criteria_block
from ..schemas import ClarifyOut, JudgeOut, RecheckOut, ReviewOut, SegmentationOut, SetReviewOut
from .base import Agent


class SegmenterAgent(Agent):
    name = "segmenter"
    title = "Нарезчик требований"
    template = "segmenter.md.j2"
    schema = SegmentationOut


@dataclass(frozen=True)
class Lens:
    name: str
    title: str
    role: str
    focus_title: str
    criteria: tuple[str, ...]
    with_tests: bool = False

    @property
    def checklist(self) -> str:
        """Что проверить обязательно — текст в prompts/lenses/<name>.md (промпты отдельно от кода)."""
        path = Path(__file__).resolve().parent.parent / "prompts" / "lenses" / f"{self.name}.md"
        return path.read_text(encoding="utf-8").rstrip("\n")


LENSES: list[Lens] = [
    Lens(
        name="analyst",
        title="Ревьюер «Аналитик»",
        role="агент-аналитик, отвечающий за качество формулировок отдельных требований",
        focus_title="качество формулировки каждого отдельного требования",
        criteria=("unambiguous", "complete", "singular", "necessary", "appropriate", "correct", "conforming", "language"),
    ),
    Lens(
        name="qa",
        title="Ревьюер «Тестировщик»",
        role="агент-тестировщик, отвечающий за проверяемость требований и критерии приёмки",
        focus_title="проверяемость требований и согласованность критериев приёмки",
        criteria=("verifiable", "validatable", "consistent", "complete"),
        with_tests=True,
    ),
    Lens(
        name="architect",
        title="Ревьюер «Архитектор»",
        role="агент-архитектор, отвечающий за интеграцию, API, данные и реализуемость",
        focus_title="интеграция, API, модель данных, обработка событий и реализуемость",
        criteria=("feasible", "appropriate", "consistent", "complete", "correct"),
    ),
    Lens(
        name="security",
        title="Ревьюер «ИБ и риски»",
        role="агент информационной безопасности и операционных рисков банка",
        focus_title="информационная безопасность, персональные и карточные данные, операционные риски",
        criteria=("risk", "complete", "correct"),
    ),
]


class ReviewerAgent(Agent):
    template = "reviewer.md.j2"
    schema = ReviewOut

    def __init__(self, llm, lens: Lens):
        super().__init__(llm)
        self.lens = lens
        self.name = lens.name
        self.title = lens.title

    def call(self, **ctx):
        return super().call(
            role=self.lens.role,
            focus_title=self.lens.focus_title,
            criteria=criteria_block(list(self.lens.criteria)),
            checklist=self.lens.checklist,
            with_tests=self.lens.with_tests,
            **ctx,
        )


class ConsistencyAgent(Agent):
    name = "consistency"
    title = "Агент согласованности набора"
    template = "consistency.md.j2"
    schema = SetReviewOut

    def call(self, **ctx):
        return super().call(
            criteria=criteria_block(
                ["consistent", "set_complete", "comprehensible", "validatable", "feasible", "ranked", "modifiable", "traceable"]
            ),
            **ctx,
        )


class JudgeAgent(Agent):
    name = "judge"
    title = "Судья-верификатор"
    template = "judge.md.j2"
    schema = JudgeOut


class ClarifierAgent(Agent):
    name = "clarifier"
    title = "Агент уточнений"
    template = "clarifier.md.j2"
    schema = ClarifyOut


class RecheckAgent(Agent):
    name = "recheck"
    title = "Агент перепроверки"
    template = "recheck.md.j2"
    schema = RecheckOut
