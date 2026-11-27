"""Генерация отчёта: Markdown (основной), JSON (машиночитаемый)."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from ..criteria import CRITERIA, SEVERITY_ICONS, SEVERITY_NAMES, criterion_name, is_set_level
from ..llm import LLMClient
from ..orchestrator import RunState
from ..schemas import KIND_NAMES
from ..version import agent_version
from .html import AGENT_NAMES, render_html

TEMPLATES = Path(__file__).resolve().parent / "templates"


_ABSENCE = re.compile(r"^(отсутств|нет\s|не\s+(предусмотрен|задан|описан|определ|указан)|не хватает)", re.I)


def is_extra_risk(f) -> bool:
    """Предложение добавить защитную меру, о которой документ молчит (rate limiting, 2FA, аудит, флаг отключения).

    Это не дефект существующей формулировки, поэтому показывается отдельно. Критичные риски и риски в конкретных
    формулировках («cardId передаётся в строке запроса») остаются среди дефектов.
    """
    return f.criterion == "risk" and f.severity != "critical" and bool(_ABSENCE.match(f.title.strip()))


def _no_html(s: str) -> str:
    """Текст из документа и ответов модели — не разметка: HTML в нём не должен исполняться в просмотрщике."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _cell(value) -> str:
    s = "" if value is None else _no_html(str(value))
    return s.replace("|", "\\|").replace("\r", "").replace("\n", "<br>").strip()


def _quote(value: str) -> str:
    return _no_html(value or "").replace("\n", " ").strip()


def _csv_safe(value) -> str:
    """Защита от CSV-инъекции: «=», «+», «-», «@» в начале ячейки Excel считает формулой."""
    s = "" if value is None else str(value)
    return "'" + s if s[:1] in ("=", "+", "-", "@", "\t", "\r") else s


def _who(value) -> str:
    """Человекочитаемое имя агента/правила: «qa» → «Тестировщик», «правила:R-UNITS» → «Правило R-UNITS»."""
    items = value if isinstance(value, list) else [value]
    out = []
    for s in items:
        s = str(s)
        out.append(f"Правило {s.split(':', 1)[1]}" if s.startswith("правила:") else AGENT_NAMES.get(s, s))
    return ", ".join(out)


def build_context(state: RunState, llm: LLMClient) -> dict:
    doc = state.doc
    reqs = {r.rid: r for r in state.requirements}
    findings = state.findings
    by_req: dict[str, list] = {r.rid: [] for r in state.requirements}
    for f in findings:
        for rid in f.requirement_ids:
            by_req.setdefault(rid, []).append(f)

    def req_status(rid: str) -> str:
        fs = [f for f in by_req.get(rid, []) if f.is_open]
        if any(f.severity == "critical" for f in fs):
            return "❌ критично"
        if fs:
            return "⚠️ есть замечания"
        return "✅ без замечаний"

    req_rows: list[dict] = []
    for r in state.requirements:
        fs = by_req.get(r.rid, [])
        req_rows.append(
            {
                "r": r,
                "status": req_status(r.rid),
                "criteria": sorted({criterion_name(f.criterion) for f in fs if f.is_open}),
                "fids": [f.fid for f in fs],
            }
        )

    # некритичные риски ИБ и эксплуатации (2FA, rate limiting, аудит…) — не дефекты формулировок по 29148 и IEEE 830:
    # отдельный подраздел, чтобы не смешивать их с дефектами требований; в подсчётах они остаются
    risk_findings = [f for f in findings if is_extra_risk(f)]
    req_findings = [f for f in findings if f.requirement_ids and not is_extra_risk(f)]
    set_findings = [f for f in findings if not f.requirement_ids and not is_extra_risk(f)]

    set_criteria = []
    for code, c in CRITERIA.items():
        if c.level != "set":
            continue
        fs = [f for f in findings if f.criterion == code and f.is_open]
        set_criteria.append(
            {
                "c": c,
                "fids": [f.fid for f in fs],
                "ok": not fs,
                "worst": min((f.severity for f in fs), key=lambda s: {"critical": 0, "major": 1, "minor": 2}[s], default=""),
            }
        )

    by_criterion = Counter(f.criterion for f in state.open_findings)
    clean_reqs = sum(1 for row in req_rows if row["status"].startswith("✅"))

    priorities = sorted([f for f in state.open_findings if f.priority > 0], key=lambda f: f.priority)
    if not priorities:
        priorities = [f for f in state.open_findings if f.severity == "critical"][:7]

    calls = llm.calls
    real = [c for c in calls if not c.cached]
    total_seconds = (state.finished_at - state.started_at).total_seconds() if state.finished_at else 0

    return {
        "doc": doc,
        "state": state,
        "reqs": reqs,
        "req_rows": req_rows,
        "req_findings": req_findings,
        "set_findings": set_findings,
        "risk_findings": risk_findings,
        "set_criteria": set_criteria,
        "by_criterion": sorted(by_criterion.items(), key=lambda kv: -kv[1]),
        "clean_reqs": clean_reqs,
        "priorities": priorities,
        "rewrites": [
            {
                "f": f,
                "req": ", ".join(f"{rid} ({reqs[rid].source_id})" if reqs[rid].source_id else rid for rid in f.requirement_ids)
                or "—",
                "was": reqs[f.requirement_ids[0]].evidence.quote
                if f.requirement_ids
                else (f.evidence[0].quote if f.evidence else ""),
            }
            for f in state.open_findings
            if f.rewrite
        ],
        "critical": [f for f in state.open_findings if f.severity == "critical"],
        "resolved": [f for f in findings if not f.is_open],
        "llm": llm,
        "calls": calls,
        "real_calls": len(real),
        "cached_calls": len(calls) - len(real),
        "tokens_in": sum(c.prompt_tokens for c in calls),
        "tokens_out": sum(c.output_tokens + c.thoughts_tokens for c in calls),
        "total_seconds": round(total_seconds),
        "agent_version": agent_version(),
        "criteria": CRITERIA,
        "SEV": SEVERITY_NAMES,
        "ICON": SEVERITY_ICONS,
        "KIND": KIND_NAMES,
        "crit_name": criterion_name,
        "is_set_level": is_set_level,
    }


def render_markdown(state: RunState, llm: LLMClient) -> str:
    env = Environment(
        loader=FileSystemLoader(TEMPLATES),
        autoescape=False,
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["cell"] = _cell
    env.filters["q"] = _quote
    env.filters["who"] = _who
    env.filters["t"] = _no_html  # свободный текст (документ, ответы модели) — без исполняемой разметки
    text = env.get_template("report.md.j2").render(**build_context(state, llm))
    # не более одной пустой строки подряд
    out, blank = [], 0
    for line in text.splitlines():
        blank = blank + 1 if not line.strip() else 0
        if blank <= 1:
            out.append(line.rstrip())
    return "\n".join(out).strip() + "\n"


def to_json(state: RunState, llm: LLMClient) -> dict:
    doc = state.doc
    return {
        "passport": {
            "document_title": doc.title,
            "file": doc.path.name,
            "format": doc.fmt,
            "sha256": doc.sha256,
            "started_at": state.started_at.isoformat(),
            "finished_at": state.finished_at.isoformat() if state.finished_at else None,
            "provider": llm.provider,
            "model": llm.model,
            "model_versions": llm.model_versions,
            "agent_version": agent_version(),
            "quotes_checked": state.quotes_checked,
            "quotes_verified": state.quotes_verified,
        },
        "verdict": {
            "ready_for_development": state.ready_for_development,
            "ready_for_testing": state.ready_for_testing,
            "summary": state.verdict.summary if state.verdict else "",
        },
        # сколько открытых дефектов покрыто уточняющими вопросами: [с вопросом, всего]
        "question_coverage": {sev: list(state.asked(sev)) for sev in ("critical", "major")},
        "requirements": [r.model_dump() for r in state.requirements],
        "findings": [f.model_dump() for f in state.findings],
        # все находки агентов и правил до судьи (с подтверждёнными цитатами): видно, что судья объединил
        # (поле candidate_ids у итоговых дефектов) и что отклонил (rejected)
        "candidates": [c.model_dump() for c in state.candidates],
        "questions": [q.model_dump() for q in state.questions],
        "tests": [t.model_dump() for t in state.tests],
        "acceptance_checks": [c.model_dump() for c in state.acceptance],
        "pair_checks": [c.model_dump() for c in state.pairs],
        "trace": [t.model_dump() for t in state.trace],
        "checklist": [c.model_dump() for c in state.checklist],
        "rejected": [r.model_dump() for r in state.rejected],
        "unverified": [f.model_dump() for f in state.unverified],
        "steps": [asdict(s) for s in state.steps],
        "llm_calls": llm.trace(),
        "errors": state.errors,
    }


def render_html_report(state: RunState, llm: LLMClient) -> str:
    return render_html(json.loads(json.dumps(to_json(state, llm), ensure_ascii=False, default=str)), state.doc)


def write_reports(state: RunState, llm: LLMClient, report_path: Path | str) -> list[Path]:
    """Отчёт .md (основной), .json (данные) и .html (интерактивный) рядом друг с другом.

    Путь можно дать с любым расширением или без него («report.html», «report»): Markdown всегда
    пишется в .md, иначе HTML-версия затёрла бы его или файл остался бы без расширения.
    """
    report_path = Path(report_path).with_suffix(".md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_markdown(state, llm), encoding="utf-8")
    data = to_json(state, llm)
    json_path = report_path.with_suffix(".json")
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    html_path = report_path.with_suffix(".html")
    html_path.write_text(render_html(json.loads(json.dumps(data, ensure_ascii=False, default=str)), state.doc), encoding="utf-8")
    return [report_path, html_path, json_path, *write_csv(state, report_path)]


def write_csv(state: RunState, report_path: Path) -> list[Path]:
    """Дефекты и вопросы в CSV (Excel/Jira): разделитель «;», UTF-8 с BOM — открывается в Excel без настроек."""
    import csv

    priority = {"critical": "Highest", "major": "High", "minor": "Low"}
    defects_path = report_path.with_name(report_path.stem + ".defects.csv")
    with defects_path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(
            [
                "ID",
                "Критичность",
                "Приоритет Jira",
                "Критерий",
                "Источник критерия",
                "Требования",
                "Заголовок",
                "Проблема",
                "Что исправить",
                "Предлагаемая формулировка",
                "Где в документе",
                "Цитаты",
                "Статус",
            ]
        )
        for f in state.findings:
            w.writerow(
                [
                    _csv_safe(v)
                    for v in [
                        f.fid,
                        SEVERITY_NAMES.get(f.severity, f.severity),
                        priority.get(f.severity, ""),
                        criterion_name(f.criterion),
                        CRITERIA[f.criterion].source if f.criterion in CRITERIA else "",
                        ", ".join(f.requirement_ids),
                        f.title,
                        f.problem,
                        f.recommendation,
                        f.rewrite,
                        "; ".join(e.location for e in f.evidence),
                        " | ".join(e.quote for e in f.evidence),
                        {"open": "открыт", "resolved": "закрыт уточнением", "changed": "изменён"}[f.status],
                    ]
                ]
            )
    questions_path = report_path.with_name(report_path.stem + ".questions.csv")
    with questions_path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(["ID", "Блокирующий", "Кому", "Вопрос", "Варианты ответа", "Зачем", "Дефекты", "Требования", "Ответ"])
        for q in state.questions:
            w.writerow(
                [
                    _csv_safe(v)
                    for v in [
                        q.qid,
                        "да" if q.blocking else "нет",
                        q.addressee,
                        q.question,
                        " / ".join(f"{i}. {o}" for i, o in enumerate(q.options, 1)),
                        q.why,
                        ", ".join(q.finding_ids),
                        ", ".join(q.requirement_ids),
                        q.answer,
                    ]
                ]
            )
    return [defects_path, questions_path]
