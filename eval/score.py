"""Оценка качества отчёта по эталонному списку дефектов.

    python eval/score.py reports/report.json [--golden eval/golden.json] [--out eval/RESULTS.md]

Дефект эталона считается найденным, если хотя бы один дефект отчёта:
- ссылается цитатами на нужные строки (lines_all — на каждую группу строк, lines_any — на любую), и
- (если задано kw) содержит ключевые слова в заголовке, описании или цитатах.
Для части дефектов задан альтернативный критерий (alt).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def finding_lines(f: dict) -> set[int]:
    lines: set[int] = set()
    for e in f.get("evidence", []):
        s, t = e.get("line_start"), e.get("line_end") or e.get("line_start")
        if s:
            lines.update(range(s, (t or s) + 1))
    return lines


def finding_text(f: dict) -> str:
    # только формулировка дефекта, без цитат: иначе ключевое слово находится в самой цитате документа
    # и дефект «засчитывается» случайно
    parts = [f.get("title", ""), f.get("problem", ""), f.get("recommendation", ""), f.get("rewrite", "")]
    return " ".join(parts)


def matches(rule: dict, f: dict) -> bool:
    lines = finding_lines(f)
    if "lines_all" in rule and not all(lines & set(group) for group in rule["lines_all"]):
        return False
    if "lines_any" in rule and not lines & set(rule["lines_any"]):
        return False
    if "kw" in rule and not re.search(rule["kw"], finding_text(f), re.I):
        return False
    return True


def score(report: dict, golden: dict) -> dict:
    findings = report["findings"] + report.get("unverified", [])
    verified = report["findings"]
    rows, matched_ids = [], set()
    for item in golden["items"]:
        hits = [f["fid"] for f in verified if matches(item, f) or ("alt" in item and matches(item["alt"], f))]
        matched_ids.update(hits)
        rows.append({"id": item["id"], "core": item["core"], "name": item["name"], "found": bool(hits), "by": hits[:4]})
    core = [r for r in rows if r["core"]]
    extra = [f for f in verified if f["fid"] not in matched_ids]
    return {
        "recall_core": sum(r["found"] for r in core) / max(1, len(core)),
        "recall_all": sum(r["found"] for r in rows) / max(1, len(rows)),
        "core_found": sum(r["found"] for r in core),
        "core_total": len(core),
        "all_found": sum(r["found"] for r in rows),
        "all_total": len(rows),
        "rows": rows,
        "extra": [{"fid": f["fid"], "title": f["title"], "severity": f["severity"]} for f in extra],
        "findings_total": len(verified),
        "quotes": (report["passport"]["quotes_verified"], report["passport"]["quotes_checked"]),
        "dropped_unverified": len(findings) - len(verified),
        "critical_asked": _asked(report, "critical"),
    }


def _asked(report: dict, severity: str) -> tuple[int, int]:
    """Открытые дефекты этой критичности, по которым задан уточняющий вопрос: (с вопросом, всего)."""
    linked = {fid for q in report["questions"] for fid in q.get("finding_ids", [])}
    of = [f for f in report["findings"] if f["severity"] == severity and f.get("status") != "resolved"]
    return sum(f["fid"] in linked for f in of), len(of)


def to_markdown(res: dict, report: dict) -> str:
    p = report["passport"]
    out = [
        "# Оценка качества по эталону",
        "",
        f"Отчёт: `{p['file']}` · версия агента: {p['agent_version']} · модель: {', '.join(p['model_versions'])}",
        "",
        f"- **Ключевые дефекты найдены:** {res['core_found']} из {res['core_total']} ({res['recall_core']:.0%})",
        f"- **Все эталонные дефекты:** {res['all_found']} из {res['all_total']} ({res['recall_all']:.0%})",
        f"- **Цитаты подтверждены:** {res['quotes'][0]} из {res['quotes'][1]}",
        f"- **Дефектов в отчёте:** {res['findings_total']}, из них вне эталона: {len(res['extra'])}",
        f"- **Критичные дефекты с уточняющим вопросом:** {res['critical_asked'][0]} из {res['critical_asked'][1]}",
        "",
        "| ID | Ключевой | Дефект эталона | Найден | Дефекты отчёта |",
        "| :--- | :---: | :--- | :---: | :--- |",
    ]
    for r in res["rows"]:
        out.append(
            f"| {r['id']} | {'да' if r['core'] else ''} | {r['name']} | {'✅' if r['found'] else '❌'} | {', '.join(r['by'])} |"
        )
    if res["extra"]:
        out += ["", "**Дефекты отчёта вне эталона** (требуют ручной оценки — часть из них дополнительные верные находки):", ""]
        out += [f"- {e['fid']} ({e['severity']}): {e['title']}" for e in res["extra"]]
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("report", type=Path)
    ap.add_argument("--golden", type=Path, default=Path(__file__).with_name("golden.json"))
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    report = json.loads(a.report.read_text(encoding="utf-8"))
    golden = json.loads(a.golden.read_text(encoding="utf-8"))
    if report["passport"]["sha256"] != golden["sha256"]:
        print("⚠ Отчёт построен по другому файлу: номера строк эталона могут не совпасть.")
    res = score(report, golden)
    md = to_markdown(res, report)
    print(md)
    if a.out:
        a.out.write_text(md, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
