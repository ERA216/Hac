"""Генерация отчёта: Markdown (основной), JSON (машиночитаемый)."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from ..criteria import CRITERIA, SEVERITY_ICONS, SEVERITY_NAMES, criterion_name, is_set_level
from ..llm import LLMClient
from ..orchestrator import RunState
from ..schemas import KIND_NAMES
from ..version import agent_version

TEMPLATES = Path(__file__).resolve().parent / "templates"


def _cell(value) -> str:
    s = "" if value is None else str(value)
    return s.replace("|", "\\|").replace("\r", "").replace("\n", "<br>").strip()


def _quote(value: str) -> str:
    return (value or "").replace("\n", " ").strip()


AGENT_NAMES = {
    "segmenter": "Нарезчик",
    "analyst": "Аналитик",
    "qa": "Тестировщик",
    "architect": "Архитектор",
    "security": "ИБ и риски",
    "consistency": "Согласованность набора",
    "judge": "Судья",
    "clarifier": "Уточнения",
    "recheck": "Перепроверка",
}


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

    req_rows = []
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

    req_findings = [f for f in findings if f.requirement_ids]
    set_findings = [f for f in findings if not f.requirement_ids]

    set_criteria = []
    for code, c in CRITERIA.items():
        if c.level != "set":
            continue
        fs = [f for f in findings if f.criterion == code and f.is_open]
        set_criteria.append({"c": c, "fids": [f.fid for f in fs], "ok": not fs,
                             "worst": min((f.severity for f in fs), key=lambda s: {"critical": 0, "major": 1, "minor": 2}[s], default="")})

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
        "set_criteria": set_criteria,
        "by_criterion": sorted(by_criterion.items(), key=lambda kv: -kv[1]),
        "clean_reqs": clean_reqs,
        "priorities": priorities,
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
        loader=FileSystemLoader(TEMPLATES), autoescape=False, undefined=StrictUndefined,
        trim_blocks=True, lstrip_blocks=True,
    )
    env.filters["cell"] = _cell
    env.filters["q"] = _quote
    env.filters["who"] = _who
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
        "requirements": [r.model_dump() for r in state.requirements],
        "findings": [f.model_dump() for f in state.findings],
        "questions": [q.model_dump() for q in state.questions],
        "tests": [t.model_dump() for t in state.tests],
        "trace": [t.model_dump() for t in state.trace],
        "checklist": [c.model_dump() for c in state.checklist],
        "rejected": [r.model_dump() for r in state.rejected],
        "unverified": [f.model_dump() for f in state.unverified],
        "steps": [asdict(s) for s in state.steps],
        "llm_calls": llm.trace(),
        "errors": state.errors,
    }


def write_reports(state: RunState, llm: LLMClient, report_path: Path) -> list[Path]:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_markdown(state, llm), encoding="utf-8")
    json_path = report_path.with_suffix(".json")
    json_path.write_text(json.dumps(to_json(state, llm), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return [report_path, json_path]
