import re
from pathlib import Path

import pytest

from revizor.document import load_document
from revizor.orchestrator import Pipeline
from revizor.report import render_markdown, write_reports
from revizor.rules import run_rules
from revizor.verify import QuoteLocator

from .fake_llm import FakeLLM

DOC = Path(__file__).resolve().parent.parent / "examples" / "requirements.md"


@pytest.fixture(scope="module")
def doc():
    return load_document(DOC)


def test_loader_sections(doc):
    assert doc.title == "Уведомление о расходах по карте"
    ln = doc.line(37)
    assert "Ф-1" in ln.text and "Функциональные требования" in ln.section


def test_pdf_loader_keeps_tables_and_ids():
    pdf = load_document(DOC.with_suffix(".pdf"))
    assert pdf.pages and pdf.pages > 1
    texts = [ln.text for ln in pdf.lines]
    assert any(re.match(r"^\s*[-*]\s+\*\*Ф-4\.\*\*", t) for t in texts)  # склеенный пункт разделён
    assert any(t.startswith("|") and "cardId" in t for t in texts)  # таблицы сохранены
    rules = {h.rule for h in run_rules(pdf, QuoteLocator(pdf))}
    assert {"R-HTTP-GET", "R-API-TYPE"} <= rules


def test_quote_locator(doc):
    loc = QuoteLocator(doc)
    assert loc.locate("Месяц определяется по дате совершения покупки в Алматы").line_start == 38
    assert loc.locate("Держатель активной карты KZT задаёт ... По умолчанию уведомление выключено.").verified
    assert not loc.locate("Порог задаётся в долларах США").verified


def test_rules_generic_findings(doc):
    rules = {h.rule for h in run_rules(doc, QuoteLocator(doc))}
    assert {"R-HTTP-GET", "R-UNITS", "R-API-TYPE", "R-API-QUERY-PII", "R-HTTP-CODE", "R-PLACEHOLDER"} <= rules


def test_pipeline_end_to_end(doc, tmp_path):
    llm = FakeLLM()
    answers = {"Q-01": "Один"}
    state = Pipeline(llm, ask_answers=lambda qs: answers).run(doc)

    assert len(state.requirements) == 7
    # цитата нарезчика, которой нет в документе, восстановлена по номеру строки
    assert "GET" in state.requirements[6].evidence.quote
    # находка с выдуманной цитатой отброшена проверкой цитат
    assert any(f.title == "Галлюцинация" for f in state.unverified)
    assert all(e.verified for f in state.findings for e in f.evidence)
    # судья отклонил первого кандидата
    assert len(state.rejected) == 1
    assert state.findings[0].fid == "D-01"
    # вопрос ссылается только на существующие дефекты
    assert state.questions[0].finding_ids == ["D-01"]
    # ответ закрыл дефект
    assert state.findings[0].status == "resolved"
    assert not state.ready_for_development

    md = render_markdown(state, llm)
    for section in ["## 1. Паспорт прогона", "## 2. Как агент нарезал требования", "## 3. Проверка каждого требования",
                    "## 4. Проверка всего набора требований", "## 5. Уточняющие вопросы", "## 7. Сводка", "## 8. Итог"]:
        assert section in md
    assert "fake-model-001" in md

    paths = write_reports(state, llm, tmp_path / "report.md")
    assert all(p.exists() for p in paths)
