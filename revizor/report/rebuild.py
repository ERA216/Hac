"""Пересборка отчёта (Markdown и HTML) из сохранённого report.json — без обращения к модели.

Нужна, когда улучшилось оформление отчёта, а данные прогона менять нельзя: паспорт (дата, модель,
версия агента, вызовы модели) и все находки берутся из report.json как есть. В паспорт добавляется
строка о том, какой версией пересобрано оформление.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from ..document import Document
from ..llm import CallRecord
from ..orchestrator import RunState, StepRecord
from ..schemas import (
    AcceptanceCheck,
    ChecklistRow,
    Finding,
    PairCheck,
    Question,
    Rejected,
    Requirement,
    TestCase,
    TraceLinkOut,
    VerdictOut,
)


class SavedRun:
    """Журнал вызовов модели из report.json — то, что рендереру нужно от LLM-клиента."""

    def __init__(self, passport: dict, calls: list[dict]):
        self.provider = passport.get("provider", "")
        self.model = passport.get("model", "")
        self.model_versions = list(passport.get("model_versions", []))
        self.calls = [CallRecord(**c) for c in calls]

    def trace(self) -> list[dict]:
        return [asdict(c) for c in self.calls]


def _dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def state_from_report(report: dict, doc: Document) -> tuple[RunState, SavedRun]:
    p = report["passport"]
    v = report.get("verdict") or {}
    state = RunState(doc=doc)
    state.started_at = _dt(p.get("started_at")) or state.started_at
    state.finished_at = _dt(p.get("finished_at"))
    state.requirements = [Requirement.model_validate(x) for x in report.get("requirements", [])]
    state.candidates = [Finding.model_validate(x) for x in report.get("candidates", [])]
    state.unverified = [Finding.model_validate(x) for x in report.get("unverified", [])]
    state.findings = [Finding.model_validate(x) for x in report.get("findings", [])]
    state.rejected = [Rejected.model_validate(x) for x in report.get("rejected", [])]
    state.questions = [Question.model_validate(x) for x in report.get("questions", [])]
    state.tests = [TestCase.model_validate(x) for x in report.get("tests", [])]
    state.acceptance = [AcceptanceCheck.model_validate(x) for x in report.get("acceptance_checks", [])]
    state.pairs = [PairCheck.model_validate(x) for x in report.get("pair_checks", [])]
    state.trace = [TraceLinkOut.model_validate(x) for x in report.get("trace", [])]
    state.checklist = [ChecklistRow.model_validate(x) for x in report.get("checklist", [])]
    if v:
        state.verdict = VerdictOut(
            ready_for_development=bool(v.get("ready_for_development")),
            ready_for_testing=bool(v.get("ready_for_testing")),
            summary=v.get("summary", ""),
        )
    state.steps = [StepRecord(**s) for s in report.get("steps", [])]
    state.errors = list(report.get("errors", []))
    state.quotes_checked = p.get("quotes_checked", 0)
    state.quotes_verified = p.get("quotes_verified", 0)
    return state, SavedRun(p, report.get("llm_calls", []))


def rebuild(report: dict, doc: Document, md_path: Path, rebuilt_by: str) -> Path:
    """Записать Markdown-отчёт с паспортом исходного прогона и пометкой о пересборке оформления."""
    from . import render_markdown

    state, run = state_from_report(report, doc)
    text = render_markdown(state, run, version=report["passport"]["agent_version"], rebuilt_by=rebuilt_by)  # type: ignore[arg-type]
    md_path.write_text(text, encoding="utf-8")
    return md_path
