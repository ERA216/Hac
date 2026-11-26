"""Интерактивный HTML-отчёт: один самодостаточный файл без внешних зависимостей.

Строится из данных отчёта (report.json) и исходного документа, поэтому HTML можно
пересобрать для любого сохранённого прогона:

    python -m revizor.report reports/report.json examples/requirements.md

Возможности: исходный документ с подсветкой проблемных строк, фильтры дефектов,
тепловая карта «требование × критерий», форма ответов на уточняющие вопросы
с выгрузкой answers.json (подаётся обратно агенту через --answers).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from ..criteria import CRITERIA, SEVERITY_NAMES
from ..document import Document, load_document
from ..schemas import KIND_NAMES

TEMPLATES = Path(__file__).resolve().parent / "templates"

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


def build_data(report: dict, doc: Document) -> dict:
    data = dict(report)
    findings = report["findings"]
    open_f = [f for f in findings if f.get("status") != "resolved"]
    data["doc_lines"] = [{"no": ln.no, "text": ln.text, "page": ln.page, "section": ln.section} for ln in doc.lines]
    data["criteria"] = {
        code: {"name": c.name, "level": c.level, "source": c.source, "definition": c.definition} for code, c in CRITERIA.items()
    }
    data["severity_names"] = SEVERITY_NAMES
    data["kind_names"] = KIND_NAMES
    data["agent_names"] = AGENT_NAMES
    data["counts"] = {
        "requirements": len(report["requirements"]),
        "open": len(open_f),
        "critical": sum(f["severity"] == "critical" for f in open_f),
        "major": sum(f["severity"] == "major" for f in open_f),
        "minor": sum(f["severity"] == "minor" for f in open_f),
        "questions": len(report["questions"]),
        "blocking": sum(bool(q["blocking"]) for q in report["questions"]),
        "resolved": len(findings) - len(open_f),
    }
    return data


def render_html(report: dict, doc: Document) -> str:
    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=False)
    payload = json.dumps(build_data(report, doc), ensure_ascii=False, default=str).replace("</", "<\\/")
    return env.get_template("report.html.j2").render(data_json=payload, title=report["passport"]["document_title"])


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) < 2:
        print("Использование: python -m revizor.report <report.json> <документ> [<report.html>]")
        return 2
    report_path, doc_path = Path(argv[0]), Path(argv[1])
    out = Path(argv[2]) if len(argv) > 2 else report_path.with_suffix(".html")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    doc = load_document(doc_path)
    if doc.sha256 != report["passport"]["sha256"]:
        print("Внимание: документ отличается от того, по которому построен отчёт (SHA-256 не совпадает).")
    out.write_text(render_html(report, doc), encoding="utf-8")
    print(f"HTML-отчёт: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
