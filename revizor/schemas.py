"""Схемы данных.

*Out-модели — контракт ответа LLM (передаются модели как JSON Schema, поэтому без
значений по умолчанию). Остальные — внутреннее состояние прогона.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .criteria import CriterionCode, Severity

RequirementKind = Literal[
    "functional", "nonfunctional", "data", "integration", "interface", "api", "acceptance", "process", "constraint"
]

KIND_NAMES = {
    "functional": "функциональное",
    "nonfunctional": "нефункциональное",
    "data": "данные",
    "integration": "интеграция",
    "interface": "UI",
    "api": "API",
    "acceptance": "критерий приёмки",
    "process": "процесс",
    "constraint": "ограничение",
}

Addressee = Literal["Аналитик", "Product owner", "Архитектор", "Разработка", "ИБ", "Тестирование", "Бизнес"]


# ----------------------------------------------------------------- ответы LLM
class EvidenceOut(BaseModel):
    quote: str = Field(description="Дословная цитата из документа, без markdown-разметки")
    line: int = Field(description="Номер строки Lxxx, с которой начинается цитата (только число)")


class RequirementOut(BaseModel):
    source_id: str = Field(description="Идентификатор в документе (Ф-1, п. 2 …) или пустая строка")
    section: str = Field(description="Раздел документа")
    kind: RequirementKind
    summary: str = Field(description="Краткая формулировка, 5–15 слов")
    quote: str = Field(description="Дословная цитата требования")
    line_start: int
    line_end: int


class SegmentationOut(BaseModel):
    document_title: str
    requirements: list[RequirementOut]


class FindingOut(BaseModel):
    requirement_ids: list[str] = Field(description="R-xx затронутых требований; пусто для дефекта набора")
    criterion: CriterionCode
    severity: Severity
    title: str = Field(description="Суть дефекта, до 12 слов")
    evidence: list[EvidenceOut] = Field(description="1–3 дословные цитаты-доказательства")
    problem: str = Field(description="Что именно не так и к чему это приведёт")
    recommendation: str = Field(description="Конкретное исправление")
    rewrite: str = Field(description="Готовая исправленная формулировка требования или пустая строка")


class TestCaseOut(BaseModel):
    requirement_ids: list[str]
    title: str
    preconditions: str
    action: str
    expected: str
    gap: str = Field(description="Что сейчас не покрыто критериями приёмки")


class ReviewOut(BaseModel):
    findings: list[FindingOut]
    suggested_tests: list[TestCaseOut]


class TraceLinkOut(BaseModel):
    requirement_id: str
    verified_by: list[str] = Field(description="R-xx критериев приёмки, проверяющих требование")
    implemented_by: list[str] = Field(description="R-xx требований API/данных/интеграции, реализующих его")
    note: str


class SetReviewOut(BaseModel):
    findings: list[FindingOut]
    trace: list[TraceLinkOut]


class MergedFindingOut(BaseModel):
    candidate_ids: list[str] = Field(description="C-xxx объединяемых кандидатов")
    criterion: CriterionCode
    severity: Severity
    title: str
    problem: str
    recommendation: str
    rewrite: str
    priority: int = Field(description="1..7 — порядок исправления в первую очередь, 0 — вне списка")


class RejectedOut(BaseModel):
    candidate_id: str
    reason: str


class ChecklistRowOut(BaseModel):
    item: str = Field(description="Пункт чек-листа как в документе")
    answer: Literal["Да", "Нет", "Не требуется", "Не определено"]
    comment: str
    candidate_ids: list[str]


class VerdictOut(BaseModel):
    ready_for_development: bool
    ready_for_testing: bool
    summary: str


class JudgeOut(BaseModel):
    findings: list[MergedFindingOut]
    rejected: list[RejectedOut]
    checklist: list[ChecklistRowOut]
    verdict: VerdictOut


class QuestionOut(BaseModel):
    addressee: Addressee
    question: str
    why: str = Field(description="Что сломается или останется спорным без ответа")
    options: list[str] = Field(description="2–4 конкретных варианта ответа")
    finding_ids: list[str]
    requirement_ids: list[str]
    blocking: bool


class ClarifyOut(BaseModel):
    questions: list[QuestionOut]


class FindingStatusOut(BaseModel):
    finding_id: str
    status: Literal["resolved", "open", "changed"]
    comment: str


class RecheckOut(BaseModel):
    statuses: list[FindingStatusOut]
    new_findings: list[FindingOut]


# ------------------------------------------------------- внутреннее состояние
class Evidence(BaseModel):
    quote: str
    line_start: int | None = None
    line_end: int | None = None
    location: str = ""
    verified: bool = False
    score: float = 0.0


class Requirement(BaseModel):
    rid: str
    source_id: str = ""
    section: str = ""
    kind: str = "functional"
    summary: str = ""
    evidence: Evidence


class Finding(BaseModel):
    fid: str
    requirement_ids: list[str] = []
    criterion: str
    severity: str
    title: str
    problem: str
    recommendation: str = ""
    rewrite: str = ""
    evidence: list[Evidence] = []
    sources: list[str] = []
    candidate_ids: list[str] = []
    priority: int = 0
    status: Literal["open", "resolved", "changed"] = "open"
    status_comment: str = ""

    @property
    def is_open(self) -> bool:
        return self.status != "resolved"


class Question(BaseModel):
    qid: str
    addressee: str
    question: str
    why: str
    options: list[str]
    finding_ids: list[str] = []
    requirement_ids: list[str] = []
    blocking: bool = False
    answer: str = ""


class TestCase(BaseModel):
    tid: str
    requirement_ids: list[str] = []
    title: str
    preconditions: str
    action: str
    expected: str
    gap: str = ""


class ChecklistRow(BaseModel):
    item: str
    answer: str
    comment: str
    finding_ids: list[str] = []


class Rejected(BaseModel):
    candidate_id: str
    title: str
    source: str
    reason: str
