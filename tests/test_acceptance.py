"""Построчная сверка критериев приёмки с требованиями."""

import json

from revizor.document import load_document
from revizor.orchestrator import Pipeline
from revizor.report import render_html_report, render_markdown, to_json

from .fake_llm import FakeLLM
from .test_pipeline import DOC


def test_contradicting_acceptance_row_becomes_critical_candidate():
    llm = FakeLLM()
    state = Pipeline(llm).run(load_document(DOC))
    assert [c.verdict for c in state.acceptance] == ["совпадает", "противоречит"]
    bad = state.acceptance[1]
    assert bad.acceptance.verified and bad.requirement.verified  # обе цитаты найдены в документе
    assert bad.acceptance.line_start == 64 and bad.requirement.line_start == 38
    # противоречие стало дефектом, а ссылка сверки ведёт на итоговый номер D-xx
    assert bad.finding_id.startswith("D-")
    finding = next(f for f in state.findings if f.fid == bad.finding_id)
    assert {e.line_start for e in finding.evidence} >= {38, 64}
    assert state.acceptance[0].finding_id == ""


def test_acceptance_checks_in_all_report_formats():
    llm = FakeLLM()
    state = Pipeline(llm).run(load_document(DOC))
    md = render_markdown(state, llm)
    assert "Сверка критериев приёмки с требованиями" in md and "❌ противоречит" in md
    data = to_json(state, llm)
    assert len(data["acceptance_checks"]) == 2
    html = render_html_report(state, llm)
    payload = html.split('<script id="data" type="application/json">', 1)[1].split("</script>", 1)[0]
    assert json.loads(payload.replace("<\\/", "</"))["acceptance_checks"][1]["verdict"] == "противоречит"
