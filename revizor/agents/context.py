"""Компактные текстовые представления состояния для промптов агентов."""

from __future__ import annotations

from ..criteria import SEVERITY_NAMES, criterion_name
from ..schemas import Finding, Question, Requirement


def _lines(ev) -> str:
    if not ev.line_start:
        return "L?"
    return f"L{ev.line_start}" if ev.line_end in (None, ev.line_start) else f"L{ev.line_start}–{ev.line_end}"


def requirements_table(reqs: list[Requirement]) -> str:
    rows = ["R-ID | ID в документе | Тип | Строки | Раздел | Краткая формулировка"]
    for r in reqs:
        rows.append(f"{r.rid} | {r.source_id or '—'} | {r.kind} | {_lines(r.evidence)} | {r.section} | {r.summary}")
    return "\n".join(rows)


def short_findings(findings: list[Finding]) -> str:
    if not findings:
        return "(пока ничего)"
    return "\n".join(
        f"- {f.fid} [{', '.join(f.sources)}] {f.title} ({', '.join(_lines(e) for e in f.evidence)})" for f in findings
    )


def full_findings(findings: list[Finding]) -> str:
    out = []
    for f in findings:
        quotes = "; ".join(f"{_lines(e)} «{e.quote}»" for e in f.evidence)
        out.append(
            f"{f.fid} [источник: {', '.join(f.sources)}] критерий={f.criterion} "
            f"({criterion_name(f.criterion)}), критичность={f.severity} ({SEVERITY_NAMES.get(f.severity, f.severity)}), "
            f"требования={', '.join(f.requirement_ids) or '—'}\n"
            f"  Заголовок: {f.title}\n  Проблема: {f.problem}\n  Рекомендация: {f.recommendation}\n"
            f"  Цитаты: {quotes}"
        )
    return "\n\n".join(out) if out else "(нет)"


def questions_answers(questions: list[Question]) -> str:
    out = []
    for q in questions:
        out.append(
            f"{q.qid} ({q.addressee}; дефекты {', '.join(q.finding_ids) or '—'}): {q.question}\n"
            f"  Ответ: {q.answer or '(нет ответа)'}"
        )
    return "\n".join(out)
