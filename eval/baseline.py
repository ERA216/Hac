"""Базовая линия «один промпт»: что даёт архитектура агента по сравнению с одним вызовом модели.

Тот же документ (построчно), те же критерии 29148 / IEEE 830, те же правила доказательности и та же схема
дефекта, та же модель — но один вызов: без нарезки, четырёх ревьюеров, детерминированных правил,
агента согласованности, судьи и программной проверки цитат. Результат оценивается тем же эталоном.

    python eval/baseline.py examples/requirements.md eval/baseline/report.json [--model gemini-3.6-flash]
    python eval/score.py eval/baseline/report.json --out eval/baseline/RESULTS.md

Ответ модели сохраняется в eval/baseline/llm-cache: с флагом --replay прогон повторяется без ключа и сети.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pydantic import BaseModel  # noqa: E402

from revizor.config import llm_from_env, load_env  # noqa: E402
from revizor.criteria import CRITERIA, SEVERITY_GUIDE, criteria_block  # noqa: E402
from revizor.document import load_document  # noqa: E402
from revizor.schemas import FindingOut  # noqa: E402
from revizor.verify import QuoteLocator  # noqa: E402


class BaselineOut(BaseModel):
    findings: list[FindingOut]


SYSTEM = """Ты — эксперт по качеству требований. Проведи статическое ревью документа требований (финтех, банк) \
до начала разработки: ищи дефекты формулировок, программу не запускай.
Документ передан построчно в формате «L012| текст». Номер строки — число после L.
Каждый дефект подкрепляй ДОСЛОВНОЙ цитатой из документа (evidence.quote) и номером строки (evidence.line). \
Нет цитаты — нет дефекта. Пиши по-русски и конкретно: что не так и к чему это приведёт; recommendation — \
конкретное действие. requirement_ids оставь пустым списком.
Шкала критичности:
{severity}
Ответ — строго JSON по заданной схеме."""

USER = """Проверь каждое требование и весь набор требований целиком по критериям:
{criteria}

Найди все дефекты: двусмысленность, непроверяемость, противоречия (в том числе между частями документа \
и с критериями приёмки), пропуски, лишние проектные решения, риски.

Документ «{title}»:
{numbered}"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path)
    ap.add_argument("out", type=Path, help="куда записать report.json (формат, который понимает eval/score.py)")
    ap.add_argument("--model")
    ap.add_argument("--fallback-models")
    ap.add_argument("--replay", action="store_true", help="только сохранённый ответ модели, без ключа и сети")
    ap.add_argument("--cache-dir", type=Path, help="каталог ответов модели (по умолчанию llm-cache рядом с out)")
    ap.add_argument("--no-cache", action="store_true", help="не брать ответ из кэша (повторный независимый прогон)")
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    load_env()
    doc = load_document(a.input)
    llm = llm_from_env(
        model=a.model,
        fallback_models=a.fallback_models,
        cache_dir=None if a.no_cache else a.cache_dir or a.out.parent / "llm-cache",
        replay=a.replay,
        max_calls=2,
        on_wait=lambda m: print("…", m, flush=True),
    )
    if not a.replay:
        llm.preflight()
    system = SYSTEM.format(severity=SEVERITY_GUIDE)
    user = USER.format(criteria=criteria_block(list(CRITERIA)), title=doc.title, numbered=doc.numbered())
    out = llm.generate("baseline", system, user, BaselineOut)

    loc = QuoteLocator(doc)
    findings, checked, verified = [], 0, 0
    for i, f in enumerate(out.findings, start=1):
        evidence = [loc.locate(e.quote, e.line) for e in f.evidence]
        checked += len(evidence)
        verified += sum(e.verified for e in evidence)
        findings.append(
            {
                "fid": f"B-{i:02d}",
                "severity": f.severity,
                "criterion": f.criterion,
                "title": f.title,
                "problem": f.problem,
                "recommendation": f.recommendation,
                "rewrite": f.rewrite,
                "evidence": [e.model_dump() for e in evidence],
            }
        )
    call = llm.calls[-1]
    report = {
        "passport": {
            "document_title": doc.title,
            "file": doc.path.name,
            "sha256": doc.sha256,
            "agent_version": "базовая линия: один промпт",
            "model_versions": [call.model_version or call.model],
            "quotes_checked": checked,
            "quotes_verified": verified,
        },
        # дефекты с неподтверждёнными цитатами не отбрасываются — базовой линии это только на руку
        "findings": findings,
        "questions": [],
        "llm_calls": llm.trace(),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    model = report["passport"]["model_versions"][0]
    print(f"Один промпт ({model}): {len(findings)} дефектов, цитат подтверждено {verified} из {checked}")
    print(f"Отчёт: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
